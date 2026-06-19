"""
Bear ASCII Face Animation Module

Provides a BearAnimator class that renders an animated ASCII bear face
in the terminal with blinking eyes, talking mouth animation, status line,
and context-aware keybind instructions.
"""

import sys
import threading
import random
import time

# ─── Data Tables ────────────────────────────────────────────────────────────

EYES = {
    "open":  ("0}", "{0"),
    "half":  ("o}", "{o"),
    "blink": ("-}", "{-"),
}

MOUTH = {
    "closed": ("  `--'  ", "        "),
    "small":  (" ( -- ) ", "        "),
    "medium": (" (    ) ", "  `--'  "),
    "open":   (" (    ) ", " ( ~~ ) "),
}

TALK_SEQ = ["small", "medium", "open", "medium", "open", "small", "closed", "closed"]

# ─── Keybind Presets ────────────────────────────────────────────────────────

KEYBINDS_IDLE = "  [R] Record  [Q] Quit"
KEYBINDS_RECORDING = "  [ENTER] Cancel  [Q] Quit"
KEYBINDS_PLAYBACK = "  [R] Interrupt & Record  [Q] Quit"
KEYBINDS_PROCESSING = "  Processing..."
KEYBIND_SEPARATOR = "─" * 35


# ─── BearAnimator ───────────────────────────────────────────────────────────

class BearAnimator:
    """Manages bear face rendering with concurrent blink and talk animations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._eye_state = "open"
        self._mouth_state = "closed"
        self._status = ""
        self._keybinds = KEYBINDS_IDLE
        self._talk_stop_event = None
        self._talk_thread = None
        self._blink_thread = None
        self._running = False

    # ── Rendering ──────────────────────────────────────────────────────────

    def _render(self):
        """Render the bear face, status, and keybinds to terminal.
        Must be called while holding self._lock."""
        le, re = EYES[self._eye_state]
        m1, m2 = MOUTH[self._mouth_state]

        # Move cursor to home and clear screen (minimal flicker)
        buf = "\033[H\033[2J"
        buf += f"""
      _,----,_
   .'  (    )  '.
  / '-.  ''  .-' \\
 | /   `----'   \\ |
 |/  ,--'  '--.  \\|
 |  / {le}  {re} \\  |
 | |    .--.    | |
  \\|   /    \\   |/
   |  |  /\\  |  |
   |   \\____/   |
   \\  {m1}  /
    '.{m2}.'
       `----'
"""
        # Status line
        if self._status:
            buf += f"\n  {self._status}\n"
        else:
            buf += "\n\n"

        # Keybind bar
        buf += f"\n{KEYBIND_SEPARATOR}\n{self._keybinds}\n"

        sys.stdout.write(buf)
        sys.stdout.flush()

    def draw(self):
        """Thread-safe draw call."""
        with self._lock:
            self._render()

    # ── Status and Keybinds ────────────────────────────────────────────────

    def set_status(self, text, keybinds=None):
        """Update the status line and optionally the keybind bar, then redraw."""
        with self._lock:
            self._status = text
            if keybinds is not None:
                self._keybinds = keybinds
            self._render()

    def set_keybinds(self, keybinds):
        """Update keybind bar and redraw."""
        with self._lock:
            self._keybinds = keybinds
            self._render()

    # ── Blink Animation ───────────────────────────────────────────────────

    def start(self):
        """Start the blink loop. Call once after all models are loaded."""
        self._running = True
        self._blink_thread = threading.Thread(target=self._blink_loop, daemon=True)
        self._blink_thread.start()
        # Initial draw
        self.draw()

    def stop(self):
        """Stop all animation threads."""
        self._running = False

    def _blink_loop(self):
        """Background blink animation — runs forever as daemon thread."""
        while self._running:
            time.sleep(random.uniform(2.4, 4.2))
            if not self._running:
                break
            for state, dur in [("half", 0.08), ("blink", 0.11), ("open", 0)]:
                with self._lock:
                    self._eye_state = state
                    self._render()
                if dur:
                    time.sleep(dur)

    # ── Talk Animation ─────────────────────────────────────────────────────

    def start_talking(self):
        """Begin mouth animation cycle. Call before TTS playback starts."""
        if self._talk_stop_event is not None:
            # Already talking, stop previous
            self._talk_stop_event.set()
            if self._talk_thread:
                self._talk_thread.join(timeout=1)

        self._talk_stop_event = threading.Event()
        self._talk_thread = threading.Thread(
            target=self._talk_loop,
            args=(self._talk_stop_event,),
            daemon=True,
        )
        self._talk_thread.start()

    def stop_talking(self):
        """Stop mouth animation. Call after TTS playback ends."""
        if self._talk_stop_event is not None:
            self._talk_stop_event.set()
            if self._talk_thread:
                self._talk_thread.join(timeout=1)
            self._talk_stop_event = None
            self._talk_thread = None

        # Reset mouth to closed
        with self._lock:
            self._mouth_state = "closed"
            self._render()

    def _talk_loop(self, stop_event):
        """Cycle through talk sequence while speech is active."""
        i = 0
        while not stop_event.is_set():
            with self._lock:
                self._mouth_state = TALK_SEQ[i % len(TALK_SEQ)]
                self._render()
            i += 1
            time.sleep(0.15)

        # Ensure mouth closes when done
        with self._lock:
            self._mouth_state = "closed"
            self._render()
