# ASCII Bear Face Animation — Instructions for AI Models

## Overview

Generate an animated ASCII bear face in a terminal or web widget. The face has:
- **Blinking eyes** on a random timer (~2–4 seconds between blinks)
- **Talking mouth** that animates open/close while speech plays
- No mood system — only speak and blink states

---

## ASCII Face Structure

The base face template (Python f-string format):

```
      _,----,_
   .'  (    )  '.
  / '-.  ''  .-' \
 | /   `----'   \ |
 |/  ,--'  '--.  \|
 |  / {le}  {re} \  |
 | |    .--.    | |
  \|   /    \   |/
   |  |  /\  |  |
   |   \____/   |
   \  {m1}  /
    '.{m2}.'
       `----`
```

- `{le}` = left eye (3 chars)
- `{re}` = right eye (3 chars)
- `{m1}` = mouth row 1 (8 chars)
- `{m2}` = mouth row 2 (8 chars)

---

## Eye States

| State  | Left `{le}` | Right `{re}` | When to use           |
|--------|------------|-------------|------------------------|
| open   | `0}`       | `{0`        | Default/idle           |
| half   | `o}`       | `{o`        | Mid-blink (transition) |
| blink  | `-}`       | `{-`        | Eyes fully closed      |

---

## Mouth States

| State  | `{m1}`       | `{m2}`       | When to use        |
|--------|--------------|--------------|--------------------|
| closed | `  \`--'  ` | `        `   | Idle / not speaking |
| small  | ` ( -- ) `  | `        `   | Slight open        |
| medium | ` (    ) `  | `  \`--'  ` | Half open          |
| open   | ` (    ) `  | ` ( ~~ ) `  | Fully open         |

---

## Blink Animation Sequence

Run in a background thread/loop. Repeat on a random interval.

```
1. eye_state = "half"   → render → wait 80ms
2. eye_state = "blink"  → render → wait 110ms
3. eye_state = "open"   → render
4. Wait random(2400ms, 4200ms) → repeat
```

---

## Talk Animation Sequence

Cycle through this list while speech is active (each frame ~150ms):

```python
TALK_SEQ = ["small", "medium", "open", "medium", "open", "small", "closed", "closed"]
```

When speech stops: set `mouth_state = "closed"` and render once.

---

## Render Method

Clear the terminal and reprint the full face each frame:

```python
import sys

def draw(eye_state, mouth_state):
    le, re = EYES[eye_state]
    m1, m2 = MOUTH[mouth_state]
    sys.stdout.write("\033[H\033[2J")   # clear screen (no flicker)
    print(f"""
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
       `----'""")
    sys.stdout.flush()
```

---

## Threading Model (Python)

Use two daemon threads — one for blinking, one for talking:

```python
import threading, random, time

lock = threading.Lock()   # prevent concurrent render corruption

# Blink thread — runs forever
def blink_loop():
    global eye_state
    while True:
        time.sleep(random.uniform(2.4, 4.2))
        for state, dur in [("half", 0.08), ("blink", 0.11), ("open", 0)]:
            with lock:
                eye_state = state
                draw(eye_state, mouth_state)
            if dur: time.sleep(dur)

# Talk thread — runs while speech is active
def talk_animation(stop_event):
    global mouth_state
    i = 0
    while not stop_event.is_set():
        with lock:
            mouth_state = TALK_SEQ[i % len(TALK_SEQ)]
            draw(eye_state, mouth_state)
        i += 1
        time.sleep(0.15)
    with lock:
        mouth_state = "closed"
        draw(eye_state, mouth_state)
```

---

## Integration with TTS

```python
threading.Thread(target=blink_loop, daemon=True).start()

def speak(text):
    stop_event = threading.Event()
    t = threading.Thread(target=talk_animation, args=(stop_event,), daemon=True)
    t.start()
    your_tts.speak(text)    # blocking TTS call
    stop_event.set()
    t.join()

speak("Hello, I am your bear assistant!")
```

Replace `your_tts.speak(text)` with your actual TTS library call (e.g. `pyttsx3`, `gTTS`, `edge-tts`, etc.).

---

## Complete Data Tables (copy-paste ready)

```python
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
```

---

## Rules Summary for AI Code Generation

1. Always use `\033[H\033[2J` to clear the terminal before each render — never `print()` new lines
2. Use a `threading.Lock()` to guard all `draw()` calls — blink and talk threads run concurrently
3. Both threads must be `daemon=True` so they die when the main program exits
4. The blink loop runs **forever** in the background — do not join it
5. The talk thread uses a `threading.Event()` stop signal — set it after TTS finishes
6. Eye state and mouth state are **independent** — blinking during speech is intentional and correct
7. Each mouth frame is 150ms; each blink transition step is 80–110ms
8. On stop: always reset `mouth_state = "closed"` and render once before the talk thread exits
