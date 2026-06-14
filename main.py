import os
import sys
import time
import wave
import subprocess
import requests
import torch
import pygame
import select
import tty
import termios
import re  # Ditambahkan untuk membersihkan tag <think>
from qwen_asr import Qwen3ASRModel
from piper.voice import PiperVoice

# Configuration
LLAMA_SERVER_PATH = "./llama-b9118/llama-server"
MODEL_PATH = "models/Qwen3.5-2B-Q4_K_M.gguf"
ASR_MODEL_PATH = "models/Qwen3-ASR-0.6B"
TTS_MODEL_PATH = "models/piper/id_ID-news_tts-medium.onnx"
PORT = 8080
API_URL = f"http://localhost:{PORT}/v1/chat/completions"

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
        self.llama_proc = None
        self.asr_model = None
        self.tts_voice = None
        # Initialize pygame mixer
        pygame.mixer.init()

    def start_llama_server(self):
        print(f"Starting llama-server with model {MODEL_PATH} on port {PORT}...")
        # Automatically determine threads
        threads = max(1, (os.cpu_count() or 4) - 1)

        # Build command with --mlock and --no-mmap to load model completely to RAM and lock it
        cmd = [
            LLAMA_SERVER_PATH,
            "-m", MODEL_PATH,
            "--port", str(PORT),
            "--mlock",
            "--no-mmap",
            "-t", str(threads),
            "-c", "2048",
            "--no-jinja"
        ]

        # Run subprocess
        self.llama_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to become ready
        ready = False
        print("Waiting for llama-server to initialize and lock weights in RAM...")
        for _ in range(30):
            try:
                resp = requests.get(f"http://localhost:{PORT}/health", timeout=1)
                if resp.status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1)

        if not ready:
            # Check if process died
            poll = self.llama_proc.poll()
            if poll is not None:
                stdout, stderr = self.llama_proc.communicate()
                print(f"llama-server failed to start (exit code {poll})")
                print("STDOUT:", stdout)
                print("STDERR:", stderr)
            raise RuntimeError("llama-server did not become ready within 30 seconds.")

        print("llama-server is ready and running!")

    def load_asr_model(self):
        print("Loading Qwen3-ASR model on CPU...")
        # Since CPU is used, we load it in float32 for maximum compatibility
        self.asr_model = Qwen3ASRModel.from_pretrained(
            ASR_MODEL_PATH,
            dtype=torch.float32,
            device_map="cpu"
        )
        print("Qwen3-ASR model loaded successfully!")

    def load_tts_model(self):
        print("Loading Piper TTS voice model...")
        self.tts_voice = PiperVoice.load(TTS_MODEL_PATH)
        print("Piper TTS loaded successfully!")

    def speech_to_text(self, audio_path):
        print(f"Running ASR on: {audio_path}")
        start_time = time.time()
        results = self.asr_model.transcribe(
            audio=audio_path,
            language="Indonesian"  # Force Indonesian language
        )
        duration = time.time() - start_time
        text = results[0].text
        print(f"ASR transcription [{duration:.2f}s]: \"{text}\"")
        return text

    def clean_llm_response(self, text):
        """
        Membersihkan respon LLM dari tag <think>...</think> beserta isinya,
        serta menghapus tag yatim piatu yang mungkin tersisa di awal/akhir teks.
        """
        # Hapus seluruh isi di dalam <think> ... </think> termasuk tag-nya sendiri
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Hapus jika ada tag <think> atau </think> yang terpisah/tersisa tanpa pasangan
        text = re.sub(r'</?think>', '', text)

        # Bersihkan whitespace berlebih di awal, akhir, dan spasi ganda di tengah teks
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def query_llm(self, user_text):
        print("Sending prompt to Qwen3.5 via llama-server...")
        system_prompt = (
            "Kamu adalah asisten AI berbasis suara.\n"
            "Gunakan Bahasa Indonesia.\n"
            "Jawab dengan jelas, ringkas, dan langsung pada intinya.\n"
            "Hindari jawaban yang terlalu panjang."
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }

        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=30)
        duration = time.time() - start_time

        response.raise_for_status()
        res_json = response.json()
        raw_response_text = res_json["choices"][0]["message"]["content"].strip()

        # Bersihkan respon dari tag <think> sebelum dicetak dan dikirim ke TTS
        response_text = self.clean_llm_response(raw_response_text)

        print(f"LLM response (Raw)  [{duration:.2f}s]: \"{raw_response_text}\"")
        print(f"LLM response (Clean) [{duration:.2f}s]: \"{response_text}\"")
        return response_text

    def text_to_speech(self, text, output_path):
        print(f"Synthesizing speech via Piper TTS to: {output_path}")
        start_time = time.time()
        with wave.open(output_path, "wb") as wav_file:
            self.tts_voice.synthesize_wav(text, wav_file)
        duration = time.time() - start_time
        print(f"TTS synthesis complete [{duration:.2f}s].")

    def run_pipeline(self, input_audio_path, output_audio_path):
        # 1. Speech-to-Text
        transcribed_text = self.speech_to_text(input_audio_path)
        if not transcribed_text.strip():
            print("Error: Transcribed text is empty.")
            return

        # 2. LLM response
        response_text = self.query_llm(transcribed_text)

        # 3. Text-to-Speech
        self.text_to_speech(response_text, output_audio_path)
        print("Pipeline executed successfully!")

    def record_audio(self, output_path, kb_input):
        print("\n[Recording] Recording started... Press ENTER to stop recording.", flush=True)
        # Use pw-record: 16kHz, 16-bit mono wav
        cmd = ["pw-record", "--channels=1", "--rate=16000", "--format=s16", output_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            while True:
                char = kb_input.get_char()
                if char in ('\n', '\r'):
                    break
                time.sleep(0.05)
        finally:
            proc.terminate()
            proc.wait()
        print("[Recording] Recording stopped.", flush=True)

    def play_audio_and_listen(self, audio_path, kb_input):
        if not os.path.exists(audio_path):
            return None

        print("\n[Playback] Playing response... Press 'R' to interrupt and record a new question, or 'Q' to quit.", flush=True)
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        interrupted_by = None
        while pygame.mixer.music.get_busy():
            char = kb_input.get_char()
            if char:
                char_lower = char.lower()
                if char_lower in ('r', 'q'):
                    interrupted_by = char_lower
                    pygame.mixer.music.stop()
                    break
            time.sleep(0.05)

        # Clear out any remaining characters in standard input buffer
        while kb_input.get_char():
            pass

        return interrupted_by

    def cleanup(self):
        if self.llama_proc:
            print("Terminating llama-server...")
            self.llama_proc.terminate()
            try:
                self.llama_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.llama_proc.kill()
            print("llama-server stopped.")

def main():
    pipeline = VoiceAssistantPipeline()
    kb = KeyboardInput()

    try:
        # Start and load everything
        pipeline.start_llama_server()
        pipeline.load_asr_model()
        pipeline.load_tts_model()

        # Define files
        input_audio = "input_record.wav"
        output_audio = "output_response.wav"

        kb.enable_raw()

        print("\n==============================================")
        print("Voice Assistant is ready!")
        print("Press 'R' to record your query.")
        print("Press 'Q' to quit the application.")
        print("==============================================\n", flush=True)

        trigger_record = False
        while True:
            if trigger_record:
                char_lower = 'r'
                trigger_record = False
            else:
                char = kb.get_char()
                char_lower = char.lower() if char else None

            if char_lower == 'q':
                break
            elif char_lower == 'r':
                # Start recording
                pipeline.record_audio(input_audio, kb)

                # Process
                print("\nProcessing ASR, LLM and TTS...", flush=True)
                pipeline.run_pipeline(input_audio, output_audio)

                # Play response and check if interrupted
                action = pipeline.play_audio_and_listen(output_audio, kb)
                if action == 'q':
                    break
                elif action == 'r':
                    trigger_record = True
                    continue
                else:
                    print("\nPress 'R' to record, 'Q' to quit.", flush=True)
            time.sleep(0.05)

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
    finally:
        kb.disable_raw()
        pipeline.cleanup()

if __name__ == "__main__":
    main()
