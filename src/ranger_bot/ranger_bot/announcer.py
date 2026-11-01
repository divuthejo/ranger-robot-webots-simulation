"""Spoken status on the laptop speakers (default), or experimentally the PS4 controller.

    /ranger/mode         -> "Manual mode" / "Shared mode" / "Auto mode"
    /ranger/task_status  -> "Going to the deer", "Arrived at the deer", "Stopped", ...
    /ranger/say          -> any text, spoken as is (voice_control: "Sorry, say that again")

Voice: Piper (offline neural TTS, models/en_US-lessac-medium.onnx), ~0.3 s per sentence.

Controller speaker (output:=controller, NOT working on this laptop - 2026-10-31 the user
heard nothing from it with the evdev driver; kept for later): over USB the DualShock 4 is also a sound card, but Linux only feeds its
headphone jack. The built-in speaker plays once its volume is set in HID output report 0x05
(flag 0x80, byte 22) - the part a PS4 normally does. That needs write access to the
controller's /dev/hidraw* (udev rule 70-ds4-hidraw.rules, plugdev). Over Bluetooth there is
no controller sound card (a PS4 streams SBC audio inside HID reports), so speech falls back
to the laptop.
"""

import glob
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import wave

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

DS4_HID_IDS = ('0003:0000054C:000009CC', '0003:0000054C:000005C4')   # DS4 v2 / v1, USB
SPEAKER_VOLUME = 0x4F
MODE_DELAY = 0.4      # s: a task status arriving this soon after a mode change replaces it


def spoken_status(text):
    """Task status line -> what to say, or None to stay quiet."""
    text = text.strip()
    if text.startswith('Not understood'):
        return "Sorry, I don't know that place" if 'no known place' in text else 'Sorry?'
    if text.startswith('Stopped'):
        return 'Stopped'
    if text.endswith('cancelled'):
        return None                 # always follows a stop / takeover / mode change, already said
    text = re.sub(r'\s*\(.*?\)', '', text)          # "(standing at 3.1, 4.2)" -> ""
    return text or None


def mode_phrase(mode):
    return {'manual': 'Manual mode', 'shared': 'Shared mode', 'auto': 'Auto mode'}.get(mode)


def find_ds4_hidraw():
    """/dev/hidrawN of a USB-connected DualShock 4, or None."""
    for uevent in glob.glob('/sys/class/hidraw/hidraw*/device/uevent'):
        try:
            with open(uevent) as f:
                if any(i in f.read().upper() for i in DS4_HID_IDS):
                    return '/dev/' + uevent.split('/')[4]
        except OSError:
            pass
    return None


def speaker_report(volume=SPEAKER_VOLUME):
    """USB output report 0x05 that only sets the speaker volume (rumble/LED untouched)."""
    r = bytearray(32)
    r[0], r[1], r[2] = 0x05, 0x80, 0x04
    r[22] = volume
    return bytes(r)


def controller_sink():
    """PipeWire sink name of the controller's sound card, or None (unplugged / Bluetooth)."""
    try:
        out = subprocess.run(['pactl', 'list', 'sinks', 'short'], capture_output=True,
                             text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        fields = line.split()
        if len(fields) > 1 and 'Wireless_Controller' in fields[1]:
            return fields[1]
    return None


class Announcer(Node):

    def __init__(self):
        super().__init__('announcer')
        model = os.path.expanduser(self.declare_parameter(
            'voice', '~/ranger_ws/models/en_US-lessac-medium.onnx').value)
        self.output = self.declare_parameter('output', 'laptop').value   # laptop|controller
        self.volume = int(self.declare_parameter('speaker_volume', SPEAKER_VOLUME).value)
        from piper import PiperVoice          # imported here so the module loads without it
        self.voice = PiperVoice.load(model)
        self.cache = {}                       # text -> wav path
        self.tmp = tempfile.mkdtemp(prefix='ranger_say_')
        self.queue = queue.Queue()
        self.pending_mode = None              # (phrase, due time)
        self.last_mode = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, 'ranger/mode', self.on_mode, latched)
        self.create_subscription(String, 'ranger/task_status', self.on_status, 10)
        self.create_subscription(String, 'ranger/say', lambda m: self.say(m.data), 10)
        self.create_timer(0.1, self.flush_mode)
        threading.Thread(target=self.speaker_loop, daemon=True).start()
        on_pad = self.output == 'controller' and controller_sink()
        self.get_logger().info(f'Announcer ready, speaking on '
                               f'{"the controller" if on_pad else "the laptop"}')

    def say(self, text):
        if text:
            self.queue.put(text)

    def on_mode(self, msg):
        first = self.last_mode is None
        if msg.data == self.last_mode:
            return
        self.last_mode = msg.data
        phrase = mode_phrase(msg.data)
        if phrase:
            self.pending_mode = (f'Ranger ready. {phrase}' if first else phrase,
                                 time.monotonic() + MODE_DELAY)

    def on_status(self, msg):
        text = spoken_status(msg.data)
        if text:
            self.pending_mode = None          # "Patrolling" says more than "Auto mode"
            self.say(text)

    def flush_mode(self):
        if self.pending_mode and time.monotonic() >= self.pending_mode[1]:
            self.say(self.pending_mode[0])
            self.pending_mode = None

    def wav_for(self, text):
        if text not in self.cache:
            path = os.path.join(self.tmp, f'{len(self.cache)}.wav')
            with wave.open(path, 'wb') as w:
                self.voice.synthesize_wav(text, w)
            self.cache[text] = path
        return self.cache[text]

    def enable_speaker(self):
        dev = find_ds4_hidraw()
        if not dev:
            return False
        try:
            with open(dev, 'wb', buffering=0) as f:
                f.write(speaker_report(self.volume))
            return True
        except OSError as e:
            self.get_logger().warn(f'Cannot enable controller speaker ({e}); install '
                                   f'70-ds4-hidraw.rules', throttle_duration_sec=60.0)
            return False

    def speaker_loop(self):
        while rclpy.ok():
            try:
                text = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            while self.queue.qsize() > 1:     # fell behind: skip to the newest lines
                text = self.queue.get()
            cmd = ['paplay']
            sink = controller_sink() if self.output == 'controller' else None
            if sink and self.enable_speaker():
                cmd.append(f'--device={sink}')
            try:
                cmd.append(self.wav_for(text))
                subprocess.run(cmd, timeout=15)
            except Exception as e:            # never let one bad line kill the voice
                self.get_logger().warn(f'Could not say "{text}": {e}')


def main(args=None):
    rclpy.init(args=args)
    node = Announcer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
