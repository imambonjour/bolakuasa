#!/usr/bin/env python3
"""Transcribe a WAV file with Qwen3-ASR 0.6B GGUF via llama-server."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import wave
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests

LLAMA_SERVER_PATH = Path("./llama-b9118/llama-server")
ASR_MODEL_PATH = Path("models/asr/Qwen3-ASR-0.6B-Q8_0.gguf")
ASR_MMPROJ_PATH = Path("models/asr/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf")
DEFAULT_PORT = 8081
DEFAULT_LANGUAGE = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV file using Qwen3-ASR 0.6B GGUF and export TXT + JSON.",
    )
    parser.add_argument("wav", type=Path, help="Input WAV file to transcribe")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path without extension (default: same basename as input WAV)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language hint for ASR (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"llama-server port for ASR (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Use an already-running ASR llama-server instead of starting one",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Leave the ASR llama-server running after transcription",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def clean_asr_response(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[-1]
    else:
        text = re.sub(r"^language\s+\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip('"').strip("'").strip()


def read_wav_metadata(wav_path: Path) -> dict:
    with wave.open(str(wav_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        duration_seconds = round(frames / sample_rate, 3) if sample_rate else 0.0
        return {
            "channels": wav_file.getnchannels(),
            "sample_rate": sample_rate,
            "sample_width_bytes": wav_file.getsampwidth(),
            "frames": frames,
            "duration_seconds": duration_seconds,
        }


def wait_for_server(proc: subprocess.Popen[str] | None, port: int, name: str, timeout_seconds: int = 60) -> None:
    for _ in range(timeout_seconds):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        if proc is not None and proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"{name} failed to start (exit code {proc.returncode}).\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        time.sleep(1)

    raise RuntimeError(f"{name} did not become ready within {timeout_seconds} seconds.")


@contextmanager
def asr_server(port: int, start_server: bool, keep_server: bool):
    proc: subprocess.Popen[str] | None = None
    if not start_server:
        wait_for_server(None, port, "ASR llama-server")
        yield
        return

    ensure_file(LLAMA_SERVER_PATH, "llama-server")
    ensure_file(ASR_MODEL_PATH, "ASR model")
    ensure_file(ASR_MMPROJ_PATH, "ASR mmproj")

    threads = max(1, (os.cpu_count() or 4) - 1)
    cmd = [
        str(LLAMA_SERVER_PATH),
        "-m",
        str(ASR_MODEL_PATH),
        "--mmproj",
        str(ASR_MMPROJ_PATH),
        "-a",
        "qwen3-asr",
        "--port",
        str(port),
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

    print(f"Starting ASR llama-server on port {port}...", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wait_for_server(proc, port, "ASR llama-server")
    print("ASR llama-server is ready.", file=sys.stderr)

    try:
        yield
    finally:
        if proc is not None and not keep_server:
            print("Stopping ASR llama-server...", file=sys.stderr)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def transcribe_with_audio_endpoint(wav_path: Path, port: int, language: str) -> tuple[str, dict]:
    transcription_url = f"http://localhost:{port}/v1/audio/transcriptions"
    with wav_path.open("rb") as audio_file:
        files = {"file": (wav_path.name, audio_file, "audio/wav")}
        data = {
            "model": "qwen3-asr",
            "language": language,
            "prompt": "Transcribe the speech. Return only the spoken text.",
        }
        response = requests.post(transcription_url, data=data, files=files, timeout=120)

    response.raise_for_status()
    payload = response.json()
    if "text" in payload:
        return payload["text"], payload
    if payload.get("choices"):
        return payload["choices"][0]["message"]["content"], payload
    raise RuntimeError(f"Unexpected ASR response: {payload}")


def transcribe_with_chat_endpoint(wav_path: Path, port: int, language: str) -> tuple[str, dict]:
    chat_url = f"http://localhost:{port}/v1/chat/completions"
    audio_data = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Transcribe this audio. Language hint: {language}. "
                            "Return only the spoken text."
                        ),
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
        "max_tokens": 512,
    }
    response = requests.post(chat_url, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"], result


def transcribe(wav_path: Path, port: int, language: str) -> tuple[str, str, dict, float]:
    start_time = time.time()
    try:
        raw_text, raw_response = transcribe_with_audio_endpoint(wav_path, port, language)
    except requests.HTTPError as exc:
        print(f"Audio transcription endpoint failed ({exc}); trying chat endpoint...", file=sys.stderr)
        raw_text, raw_response = transcribe_with_chat_endpoint(wav_path, port, language)

    text = clean_asr_response(raw_text)
    elapsed = round(time.time() - start_time, 3)
    return text, raw_text, raw_response, elapsed


def write_outputs(
    output_base: Path,
    text: str,
    wav_path: Path,
    language: str,
    raw_text: str,
    raw_response: dict,
    transcription_seconds: float,
) -> tuple[Path, Path]:
    txt_path = output_base.with_suffix(".txt")
    json_path = output_base.with_suffix(".json")

    txt_path.write_text(text + ("\n" if text else ""), encoding="utf-8")

    payload = {
        "text": text,
        "source_audio": str(wav_path.resolve()),
        "language": language,
        "model": {
            "name": "Qwen3-ASR-0.6B",
            "format": "gguf",
            "weights": str(ASR_MODEL_PATH),
            "mmproj": str(ASR_MMPROJ_PATH),
        },
        "audio": read_wav_metadata(wav_path),
        "transcription_seconds": transcription_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_text": raw_text,
        "raw_response": raw_response,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return txt_path, json_path


def main() -> int:
    args = parse_args()
    wav_path = args.wav.expanduser().resolve()
    ensure_file(wav_path, "Input WAV")

    output_base = args.output.expanduser() if args.output else wav_path.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    try:
        with asr_server(args.port, start_server=not args.no_start_server, keep_server=args.keep_server):
            print(f"Transcribing {wav_path}...", file=sys.stderr)
            text, raw_text, raw_response, elapsed = transcribe(wav_path, args.port, args.language)
            txt_path, json_path = write_outputs(
                output_base,
                text,
                wav_path,
                args.language,
                raw_text,
                raw_response,
                elapsed,
            )
    except (FileNotFoundError, RuntimeError, requests.RequestException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(text)
    print(f"\nWrote {txt_path}", file=sys.stderr)
    print(f"Wrote {json_path}", file=sys.stderr)
    print(f"Done in {elapsed:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
