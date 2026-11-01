"""Voice commands, push-to-talk: hold Options on the PS4 controller, speak, release.

    "manual" | "shared" | "auto"                 switch drive mode (/ranger/set_mode)
    "slow down" | "speed up" | "max speed" | "minimum speed"    speed level (/ranger/speed)
    anything else -> /ranger/task, e.g.          (tasks.py does the work, same as the CLI)
        "go towards the deer", "go near the pond", "check the rabbit",
        "go home", "patrol", "stop"

Offline Vosk (small English model) restricted to phrases built from the task targets, so
the few words it must tell apart are recognised reliably. Audio comes from PipeWire's
`parec` (no PortAudio). The microphone is the laptop's built-in one by default: with the
PS4 controller plugged in, the *default* input becomes the controller's headset jack,
which is silent without a headset.

Recording starts the moment Options goes down (the desktop mic icon appears) - speak
straight away, release when done (max 10 s).

Rumble: a short tick when it starts listening, a low buzz if it did not understand
(mode changes still give mode_manager's 1/2/3 pulses). What it heard: /ranger/voice.
"""

import json
import os
import queue
import subprocess
import threading
import time
import wave

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy, JoyFeedback
from std_msgs.msg import String

from ranger_bot.tasks import ALIASES, PLACES, parse_animals, parse_task

PTT_BUTTON = 9            # Options (see config/ps4_teleop.yaml)
MODE_WORDS = {'manual': 'manual', 'shared': 'shared', 'auto': 'auto', 'automatic': 'auto'}
SPEED_WORDS = {'slow down': 'slower', 'slower': 'slower', 'go slower': 'slower',
               'speed up': 'faster', 'faster': 'faster', 'go faster': 'faster',
               'max speed': 'max', 'full speed': 'max', 'top speed': 'max',
               'minimum speed': 'min', 'slowest': 'min'}
VERBS = ('go to the', 'go towards the', 'go near the', 'check the', 'check on the',
         'follow the', 'watch the')


def target_words(targets):
    """Spoken names of the targets: the US model knows 'center', and pond/water mean the lake."""
    names = {('center' if t == 'centre' else t) for t in targets}
    names |= {a for a, t in ALIASES.items() if t in targets and a not in ('start', 'base')}
    return sorted(names - {'home'})


def grammar(targets):
    """Every phrase Vosk may return; '[unk]' soaks up everything else."""
    # Several ways to say "home": the small model sometimes drops it after a lone "go".
    phrases = list(MODE_WORDS) + list(SPEED_WORDS) + ['stop', 'patrol', 'go home', 'go back home', 'return home',
                                  'come home']
    phrases += [f'{verb} {name}' for name in target_words(targets) for verb in VERBS]
    phrases += ['find the intruder', 'search for the intruder', 'follow the intruder',
                'look for the intruder']
    return phrases + ['[unk]']


def interpret(text, targets):
    """Recognised text -> ('mode', m) | ('speed', s) | ('task', text) | (None, reason)."""
    words = [w for w in text.split() if w != '[unk]']
    if not words:
        return None, 'nothing recognised'
    if len(words) == 1 and words[0] in MODE_WORDS:
        return 'mode', MODE_WORDS[words[0]]
    if ' '.join(words) in SPEED_WORDS:
        return 'speed', SPEED_WORDS[' '.join(words)]
    action, why = parse_task(' '.join(words), targets)
    if action is None:
        return None, why
    return 'task', ' '.join(words)


def pick_microphone():
    """The laptop's built-in mic: a 'Digital Microphone', else any real input that is not
    a monitor or the PS4 controller. None = PipeWire's default."""
    try:
        out = subprocess.run(['pactl', 'list', 'sources'], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    sources, name = [], None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('Name:'):
            name = line.split(':', 1)[1].strip()
        elif line.startswith('Description:') and name:
            sources.append((name, line.split(':', 1)[1].strip()))
            name = None
    real = [(n, d) for n, d in sources if '.monitor' not in n
            and 'Wireless_Controller' not in n and 'DualShock' not in d]
    for n, d in real:
        if 'Digital Microphone' in d:
            return n
    return real[0][0] if real else None


def clip_info(pcm):
    """'2.3 s, peak 41%' - length and loudness of a 16 kHz s16le recording."""
    n = len(pcm) // 2
    peak = max((abs(v) for v in memoryview(bytes(pcm[:n * 2])).cast('h')), default=0)
    return f'{n / 16000:.1f} s, peak {100 * peak / 32768:.0f}%'


class Listener:
    """Streams `parec` audio into a Vosk recognizer between start() and stop()."""

    def __init__(self, model_path, phrases, device=None, max_seconds=10.0, save_path=None):
        import vosk                       # imported here so the module loads without it
        vosk.SetLogLevel(-1)
        self.vosk = vosk
        self.model = vosk.Model(model_path)
        self.grammar = json.dumps(phrases)
        self.device = device
        self.max_seconds = max_seconds
        self.save_path = save_path
        self.proc = None
        self.results = queue.Queue()

    def recognize(self, pcm):
        """Recognise a complete 16 kHz mono s16le recording (used by tests)."""
        rec = self.vosk.KaldiRecognizer(self.model, 16000, self.grammar)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get('text', '')

    def start(self):
        if self.proc is not None:
            return
        cmd = ['parec', '--format=s16le', '--rate=16000', '--channels=1', '--raw',
               '--latency-msec=50']
        if self.device:
            cmd.append(f'--device={self.device}')
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._run, args=(self.proc,), daemon=True).start()

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc = None

    def _run(self, proc):
        rec = self.vosk.KaldiRecognizer(self.model, 16000, self.grammar)
        end = time.monotonic() + self.max_seconds
        pcm = bytearray()
        while time.monotonic() < end:
            chunk = proc.stdout.read(1600)          # 0.05 s
            if not chunk:
                break                               # released: parec terminated
            pcm += chunk
            rec.AcceptWaveform(chunk)
        proc.terminate()
        self.save(pcm)
        self.results.put((json.loads(rec.FinalResult()).get('text', ''), clip_info(pcm)))

    def save(self, pcm):
        """Keep the last utterance (16 kHz mono WAV) so a miss can be listened to."""
        if not self.save_path:
            return
        try:
            with wave.open(self.save_path, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(bytes(pcm))
        except OSError:
            pass


class VoiceControl(Node):

    def __init__(self):
        super().__init__('voice_control')
        model = os.path.expanduser(self.declare_parameter(
            'model', '~/ranger_ws/models/vosk-model-small-en-us-0.15').value)
        world = os.path.expanduser(self.declare_parameter(
            'world', '~/ranger_ws/src/ranger_bot/worlds/nature_reserve.wbt').value)
        device = self.declare_parameter('audio_device', '').value or pick_microphone()
        self.targets = dict(PLACES)
        try:
            with open(world) as f:
                self.targets.update(parse_animals(f.read()))
        except OSError as e:
            self.get_logger().warn(f'No animals loaded ({e})')
        save = os.path.expanduser(self.declare_parameter(
            'save_last', '~/.ros/voice_last.wav').value)
        self.listener = Listener(model, grammar(self.targets), device, save_path=save)
        self.mode_pub = self.create_publisher(String, 'ranger/set_mode', 10)
        self.task_pub = self.create_publisher(String, 'ranger/task', 10)
        self.speed_pub = self.create_publisher(String, 'ranger/speed', 10)
        self.heard_pub = self.create_publisher(String, 'ranger/voice', 10)
        self.rumble_pub = self.create_publisher(JoyFeedback, 'joy/set_feedback', 10)
        self.say_pub = self.create_publisher(String, 'ranger/say', 10)     # announcer.py
        self.ptt_down = False
        self.rumble_off_at = None
        self.create_subscription(Joy, 'joy', self.on_joy, 10)
        self.create_timer(0.05, self.poll)
        self.get_logger().info(
            f'Voice control ready (mic: {device or "PipeWire default"}). Hold Options and say '
            f'"auto", "go towards the deer", "go near the pond", "patrol", "stop"...')

    def rumble(self, intensity, seconds):
        self.rumble_pub.publish(JoyFeedback(type=JoyFeedback.TYPE_RUMBLE, id=0,
                                            intensity=float(intensity)))
        self.rumble_off_at = time.monotonic() + seconds

    def on_joy(self, msg):
        down = len(msg.buttons) > PTT_BUTTON and bool(msg.buttons[PTT_BUTTON])
        if down and not self.ptt_down:
            self.listener.start()
            self.rumble(0.7, 0.15)                       # tick: listening (as strong as
                                                         # a mode pulse; 0.35/0.08 s was too weak to feel)
            self.get_logger().info('Listening...')
        elif not down and self.ptt_down:
            self.listener.stop()
        self.ptt_down = down

    def poll(self):
        if self.rumble_off_at and time.monotonic() >= self.rumble_off_at:
            self.rumble_pub.publish(JoyFeedback(type=JoyFeedback.TYPE_RUMBLE, id=0, intensity=0.0))
            self.rumble_off_at = None
        try:
            text, info = self.listener.results.get_nowait()
        except queue.Empty:
            return
        self.heard_pub.publish(String(data=text))
        kind, arg = interpret(text, self.targets)
        if kind is None:
            self.get_logger().warn(f'Heard "{text}" - {arg} ({info})')
            self.rumble(0.4, 0.5)                        # low buzz: not understood
            self.say_pub.publish(String(data='Sorry, say that again'))
        elif kind == 'speed':
            self.get_logger().info(f'Heard "{text}" -> speed {arg}')
            self.speed_pub.publish(String(data=arg))
        elif kind == 'mode':
            self.get_logger().info(f'Heard "{text}" -> mode {arg}')
            self.mode_pub.publish(String(data=arg))
        else:
            self.get_logger().info(f'Heard "{text}" -> task')
            self.task_pub.publish(String(data=arg))


def main(args=None):
    rclpy.init(args=args)
    node = VoiceControl()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.listener.stop()
        node.destroy_node()
        rclpy.try_shutdown()
