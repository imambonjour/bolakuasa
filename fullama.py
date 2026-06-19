import base64
import logging
import os
import re
import select
import subprocess
import sys
import termios
import time
import tty
import wave

import numpy as np
import onnxruntime as ort
import pygame
import requests
from bear_face import (
    BearAnimator,
    KEYBINDS_IDLE,
    KEYBINDS_PLAYBACK,
    KEYBINDS_PROCESSING,
    KEYBINDS_RECORDING,
)
from piper.voice import PiperVoice


logging.basicConfig(
    filename="voice_assistant.log",
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice_assistant_gguf")

LLAMA_SERVER_PATH = "./llama-b9118/llama-server"
LLM_MODEL_PATH = "models/Qwen3.5-2B-Q4_K_M.gguf"
ASR_MODEL_PATH = "models/asr/Qwen3-ASR-0.6B-Q8_0.gguf"
ASR_MMPROJ_PATH = "models/asr/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
TTS_MODEL_PATH = "models/piper/id_ID-news_tts-medium.onnx"
VAD_MODEL_PATH = "models/silero_vad.onnx"
LLM_PORT = 8080
ASR_PORT = 8081
LLM_API_URL = f"http://localhost:{LLM_PORT}/v1/chat/completions"
ASR_TRANSCRIPTION_URL = f"http://localhost:{ASR_PORT}/v1/audio/transcriptions"
ASR_CHAT_URL = f"http://localhost:{ASR_PORT}/v1/chat/completions"
SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 30
VAD_MIN_SPEECH_MS = 250


class SileroVAD:
    CONTEXT_SIZE = 64

    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.reset_states()

    def reset_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SIZE), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def __call__(self, chunk):
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        x = np.concatenate([self._context, chunk], axis=1)
        out, state = self.session.run(
            None,
            {"input": x, "state": self._state, "sr": self._sr},
        )
        self._state = state
        self._context = x[:, -self.CONTEXT_SIZE :]
        return float(out[0, 0])


class StreamingVADIterator:
    def __init__(
        self,
        model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    ):
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = round(sampling_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = round(sampling_rate * speech_pad_ms / 1000)
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def process(self, chunk):
        window_size_samples = len(chunk)
        self.current_sample += window_size_samples
        speech_prob = self.model(chunk)

        if speech_prob >= self.threshold and self.temp_end:
            self.temp_end = 0

        if speech_prob >= self.threshold and not self.triggered:
            self.triggered = True
            speech_start = max(
                0,
                self.current_sample - self.speech_pad_samples - window_size_samples,
            )
            return {"start": int(speech_start)}

        if speech_prob < self.threshold - 0.15 and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
            self.temp_end = 0
            self.triggered = False
            return {"end": int(speech_end)}

        return None


class KeyboardInput:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def enable_raw(self):
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def disable_raw(self):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            self.old_settings = None

    def get_char(self):
        if sys.stdin.isatty():
            if select.select([sys.stdin], [], [], 0.02)[0]:
                return sys.stdin.read(1)
        return None


class VoiceAssistantPipeline:
    def __init__(self):
        self.llm_proc = None
        self.asr_proc = None
        self.tts_voice = None
        self.vad = None
        self.vad_iterator = None
        self.bear = BearAnimator()
        pygame.mixer.init()

    def _ensure_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    def _wait_for_server(self, proc, port, name, timeout_seconds=60):
        log.info(f"Waiting for {name} on port {port}...")
        for _ in range(timeout_seconds):
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                if resp.status_code == 200:
                    log.info(f"{name} is ready.")
                    return
            except requests.RequestException:
                pass
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                log.error(f"{name} failed to start (exit code {proc.returncode})")
                log.error(f"STDOUT: {stdout}")
                log.error(f"STDERR: {stderr}")
                raise RuntimeError(f"{name} failed to start.")
            time.sleep(1)
        raise RuntimeError(f"{name} did not become ready within {timeout_seconds} seconds.")

    def _start_server(self, cmd, port, name):
        log.info(f"Starting {name}: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._wait_for_server(proc, port, name)
        return proc

    def start_llm_server(self):
        self._ensure_file(LLAMA_SERVER_PATH)
        self._ensure_file(LLM_MODEL_PATH)
        threads = max(1, (os.cpu_count() or 4) - 1)
        cmd = [
            LLAMA_SERVER_PATH,
            "-m",
            LLM_MODEL_PATH,
            "--port",
            str(LLM_PORT),
            "--mlock",
            "--no-mmap",
            "-t",
            str(threads),
            "-c",
            "2048",
            "--no-jinja",
        ]
        self.llm_proc = self._start_server(cmd, LLM_PORT, "LLM llama-server")

    def start_asr_server(self):
        self._ensure_file(LLAMA_SERVER_PATH)
        self._ensure_file(ASR_MODEL_PATH)
        self._ensure_file(ASR_MMPROJ_PATH)
        threads = max(1, (os.cpu_count() or 4) - 1)
        cmd = [
            LLAMA_SERVER_PATH,
            "-m",
            ASR_MODEL_PATH,
            "--mmproj",
            ASR_MMPROJ_PATH,
            "-a",
            "qwen3-asr",
            "--port",
            str(ASR_PORT),
            "--media-path",
            ".",
            "--mlock",
            "--no-mmap",
            "-t",
            str(threads),
            "-c",
            "2048",
            "--no-webui",
        ]
        self.asr_proc = self._start_server(cmd, ASR_PORT, "ASR llama-server")

    def load_tts_model(self):
        log.info("Loading Piper TTS voice model...")
        self.tts_voice = PiperVoice.load(TTS_MODEL_PATH)
        log.info("Piper TTS loaded successfully.")

    def load_vad(self):
        log.info("Loading Silero VAD model...")
        self.vad = SileroVAD(VAD_MODEL_PATH)
        self.vad_iterator = StreamingVADIterator(self.vad)
        log.info("Silero VAD loaded successfully.")

    def _clean_asr_response(self, text):
        text = re.sub(r"<\|[^>]+?\|>", "", text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"</?think>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.strip('"').strip("'").strip()

    def _transcribe_with_audio_endpoint(self, audio_path):
        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
            data = {
                "model": "qwen3-asr",
                "language": "id",
                "prompt": "Transcribe Indonesian speech. Return only the spoken text.",
            }
            response = requests.post(
                ASR_TRANSCRIPTION_URL,
                data=data,
                files=files,
                timeout=120,
            )
        response.raise_for_status()
        payload = response.json()
        if "text" in payload:
            return payload["text"]
        if payload.get("choices"):
            return payload["choices"][0]["message"]["content"]
        raise RuntimeError(f"Unexpected ASR response: {payload}")

    def _transcribe_with_chat_endpoint(self, audio_path):
        with open(audio_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("ascii")
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this Indonesian audio. Return only the spoken text.",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 256,
        }
        response = requests.post(ASR_CHAT_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def speech_to_text(self, audio_path):
        log.info(f"Running GGUF ASR on: {audio_path}")
        start_time = time.time()
        try:
            text = self._transcribe_with_audio_endpoint(audio_path)
        except requests.HTTPError as exc:
            log.warning(f"ASR transcription endpoint failed, trying chat audio: {exc}")
            text = self._transcribe_with_chat_endpoint(audio_path)
        duration = time.time() - start_time
        text = self._clean_asr_response(text)
        log.info(f'ASR transcription [{duration:.2f}s]: "{text}"')
        return text

    def clean_llm_response(self, text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"</?think>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def query_llm(self, user_text):
        log.info("Sending prompt to Qwen3.5 via llama-server...")
        system_prompt = (
            "Kamu adalah asisten AI berbasis suara.\n"
            "Gunakan Bahasa Indonesia.\n"
            "Jawab dengan jelas, ringkas, dan langsung pada intinya.\n"
            "Hindari jawaban yang terlalu panjang."
        )
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.7,
            "max_tokens": 150,
        }
        start_time = time.time()
        response = requests.post(LLM_API_URL, json=payload, timeout=30)
        duration = time.time() - start_time
        response.raise_for_status()
        raw_response_text = response.json()["choices"][0]["message"]["content"].strip()
        response_text = self.clean_llm_response(raw_response_text)
        log.info(f'LLM response (Raw)  [{duration:.2f}s]: "{raw_response_text}"')
        log.info(f'LLM response (Clean) [{duration:.2f}s]: "{response_text}"')
        return response_text

    def text_to_speech(self, text, output_path):
        log.info(f"Synthesizing speech via Piper TTS to: {output_path}")
        start_time = time.time()
        with wave.open(output_path, "wb") as wav_file:
            self.tts_voice.synthesize_wav(text, wav_file)
        duration = time.time() - start_time
        log.info(f"TTS synthesis complete [{duration:.2f}s].")

    def run_pipeline(self, input_audio_path, output_audio_path):
        self.bear.set_status("Transcribing...", KEYBINDS_PROCESSING)
        transcribed_text = self.speech_to_text(input_audio_path)
        if not transcribed_text.strip():
            log.warning("Transcribed text is empty.")
            return False

        self.bear.set_status("Thinking...", KEYBINDS_PROCESSING)
        response_text = self.query_llm(transcribed_text)

        self.bear.set_status("Generating speech...", KEYBINDS_PROCESSING)
        self.text_to_speech(response_text, output_audio_path)
        log.info("Pipeline executed successfully.")
        return True

    def record_audio(self, output_path, kb_input):
        self.bear.set_status("Listening...", KEYBINDS_RECORDING)
        log.info("Recording started, listening for speech...")

        cmd = ["pw-record", "--channels=1", "--rate", str(SAMPLE_RATE), "--format=s16", "-a", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.vad_iterator.reset_states()

        bytes_per_window = VAD_WINDOW_SAMPLES * 2
        audio_chunks = []
        speech_start_sample = None
        speech_end_sample = None
        quit_requested = False
        cancelled = False

        try:
            while True:
                char = kb_input.get_char()
                if char in ("\n", "\r"):
                    cancelled = True
                    break
                if char and char.lower() == "q":
                    quit_requested = True
                    break

                chunk_bytes = proc.stdout.read(bytes_per_window)
                if not chunk_bytes:
                    break
                if len(chunk_bytes) < bytes_per_window:
                    chunk_bytes += b"\x00" * (bytes_per_window - len(chunk_bytes))

                samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_chunks.append(samples)

                result = self.vad_iterator.process(samples)
                if result and "start" in result:
                    speech_start_sample = result["start"]
                    self.bear.set_status("Speech detected...", KEYBINDS_RECORDING)
                    log.info("Speech detected.")
                if result and "end" in result:
                    speech_end_sample = result["end"]
                    log.info("Speech ended. Submitting...")
                    break
        finally:
            proc.terminate()
            proc.wait()

        if quit_requested:
            return "quit"
        if cancelled:
            log.info("Recording cancelled by user.")
            return None
        if speech_start_sample is None or speech_end_sample is None:
            log.info("No speech detected.")
            self.bear.set_status("No speech detected.", KEYBINDS_IDLE)
            time.sleep(1)
            return None

        full_audio = np.concatenate(audio_chunks)
        trimmed = full_audio[speech_start_sample:speech_end_sample]
        min_samples = round(SAMPLE_RATE * VAD_MIN_SPEECH_MS / 1000)
        if len(trimmed) < min_samples:
            log.info("Speech too short, ignored.")
            self.bear.set_status("Speech too short.", KEYBINDS_IDLE)
            time.sleep(1)
            return None

        pcm = np.clip(trimmed * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())

        log.info("Captured speech successfully.")
        return "ok"

    def play_audio_and_listen(self, audio_path, kb_input):
        if not os.path.exists(audio_path):
            return None

        self.bear.set_status("Speaking...", KEYBINDS_PLAYBACK)
        self.bear.start_talking()

        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        log.info("Playing response audio...")

        interrupted_by = None
        while pygame.mixer.music.get_busy():
            char = kb_input.get_char()
            if char:
                char_lower = char.lower()
                if char_lower in ("r", "q"):
                    interrupted_by = char_lower
                    pygame.mixer.music.stop()
                    break
            time.sleep(0.05)

        self.bear.stop_talking()
        while kb_input.get_char():
            pass
        return interrupted_by

    def cleanup(self):
        self.bear.stop()
        for name, proc in (("ASR llama-server", self.asr_proc), ("LLM llama-server", self.llm_proc)):
            if proc:
                log.info(f"Terminating {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.info(f"{name} stopped.")


def _print_loading(step, total, message):
    bar_len = 20
    filled = int(bar_len * step / total)
    bar = "#" * filled + "." * (bar_len - filled)
    sys.stdout.write(f"\r  [{bar}] ({step}/{total}) {message}")
    sys.stdout.flush()


def main():
    pipeline = VoiceAssistantPipeline()
    kb = KeyboardInput()

    try:
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.write("\n  Voice Assistant GGUF - Loading...\n\n")
        sys.stdout.flush()

        _print_loading(1, 4, "Starting LLM server...")
        pipeline.start_llm_server()

        _print_loading(2, 4, "Starting ASR server...")
        pipeline.start_asr_server()

        _print_loading(3, 4, "Loading TTS model...")
        pipeline.load_tts_model()

        _print_loading(4, 4, "Loading VAD model...")
        pipeline.load_vad()

        sys.stdout.write("\n\n  All models loaded.\n")
        sys.stdout.flush()
        time.sleep(0.5)

        input_audio = "input_record.wav"
        output_audio = "output_response.wav"

        kb.enable_raw()
        pipeline.bear.set_status("", KEYBINDS_IDLE)
        pipeline.bear.start()

        trigger_record = False
        while True:
            if trigger_record:
                char_lower = "r"
                trigger_record = False
            else:
                char = kb.get_char()
                char_lower = char.lower() if char else None

            if char_lower == "q":
                break
            if char_lower == "r":
                record_result = pipeline.record_audio(input_audio, kb)
                if record_result == "quit":
                    break
                if record_result != "ok":
                    pipeline.bear.set_status("", KEYBINDS_IDLE)
                    continue

                success = pipeline.run_pipeline(input_audio, output_audio)
                if not success:
                    pipeline.bear.set_status("Could not process.", KEYBINDS_IDLE)
                    time.sleep(1)
                    pipeline.bear.set_status("", KEYBINDS_IDLE)
                    continue

                action = pipeline.play_audio_and_listen(output_audio, kb)
                if action == "q":
                    break
                if action == "r":
                    trigger_record = True
                    continue
                pipeline.bear.set_status("", KEYBINDS_IDLE)
            time.sleep(0.05)

    except Exception as e:
        log.error(f"An error occurred: {e}", exc_info=True)
        print(f"\nFatal error: {e}", file=sys.stderr)
    finally:
        kb.disable_raw()
        pipeline.cleanup()


if __name__ == "__main__":
    main()
