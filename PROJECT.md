# Voice AI Assistant - Core Architecture

## Objective

Membangun chatbot berbasis suara yang dapat menerima pertanyaan dalam Bahasa Indonesia dan memberikan jawaban dalam bentuk suara.

Pipeline utama:

```text
Audio Input
    ↓
Qwen3-ASR
    ↓
Text
    ↓
Conversation Manager
    ↓
Qwen 3.5 1B (llama.cpp)
    ↓
Response Text
    ↓
Piper TTS
    ↓
Audio Output
```

---

# Core Components

## 1. Audio Input

### Role

Menerima audio dari pengguna.

### Input

```text
User Speech
```

### Output

```text
Audio Stream
```

### Notes

- Audio dikirim ke backend secara streaming atau per-request.
- Bahasa utama: Indonesia.

---

## 2. Qwen3-ASR

### Role

Speech-to-Text (STT).

Mengubah audio pengguna menjadi teks.

### Input

```text
Audio
```

### Output

```text
"Jelaskan hukum Newton pertama."
```

### Responsibilities

- Transkripsi Bahasa Indonesia.
- Mendukung percakapan sehari-hari.
- Menangani campuran Indonesia-Inggris jika memungkinkan.

---

## 3. Conversation Manager

### Role

Mengelola konteks percakapan dan prompt.

### Responsibilities

- Menyimpan riwayat percakapan.
- Menyusun prompt untuk LLM.
- Mengatur jumlah pesan yang dikirim ke model.
- Menjaga konteks tetap relevan.

### Input

```text
User Message
```

### Output

```text
Formatted Prompt
```

---

## 4. Qwen 3.5 1B (llama.cpp)

### Role

Large Language Model.

Menghasilkan respons berdasarkan prompt.

### Input

```text
Prompt
```

### Output

```text
"Hukum Newton pertama menyatakan bahwa benda akan mempertahankan keadaan diam atau bergerak lurus beraturan jika tidak ada gaya resultan yang bekerja padanya."
```

### Responsibilities

- Menjawab pertanyaan.
- Menjelaskan konsep.
- Melakukan percakapan umum.
- Mengikuti instruksi system prompt.

---

## 5. Piper TTS

### Role

Text-to-Speech.

Mengubah respons LLM menjadi audio.

### Input

```text
Response Text
```

### Output

```text
Audio Response
```

### Responsibilities

- Sintesis suara.
- Latensi rendah.
- Konsisten untuk demo.

---

# Data Flow

## Step 1

User berbicara.

```text
"Siapa presiden pertama Indonesia?"
```

↓

## Step 2

Qwen3-ASR melakukan transkripsi.

```text
Siapa presiden pertama Indonesia?
```

↓

## Step 3

Conversation Manager membangun prompt.

```text
System Prompt
+
Conversation History
+
Current User Message
```

↓

## Step 4

Qwen 3.5 1B menghasilkan jawaban.

```text
Presiden pertama Indonesia adalah Soekarno.
```

↓

## Step 5

Piper mengubah teks menjadi suara.

```text
Audio Output
```

↓

## Step 6

Audio diputar ke pengguna.

---

# Memory Strategy

## Short-Term Memory

Simpan:

```text
5-10 pesan terakhir
```

Contoh:

```text
User
Assistant
User
Assistant
User
```

Tujuan:

- Menjaga konteks percakapan.
- Mengurangi penggunaan token.
- Mempercepat inferensi.

---

# System Prompt

```text
Kamu adalah asisten AI berbasis suara.

Gunakan Bahasa Indonesia.

Jawab dengan jelas, ringkas, dan mudah dipahami.

Jika pertanyaan tidak jelas, minta klarifikasi.

Hindari jawaban yang terlalu panjang.
```

---

# Initial Technical Stack

## Backend

- Python
- FastAPI
- WebSocket

## STT

- Qwen3-ASR

## LLM

- Qwen 3.5 1B
- llama.cpp

## TTS

- Piper

## Communication

- WebSocket (real-time)
- JSON message protocol

---

# MVP Scope

Fitur yang wajib ada:

- Audio → Text
- Text → LLM
- LLM → Audio
- Context memory sederhana
- Bahasa Indonesia

Belum termasuk:

- Avatar
- Emotion detection
- Tool calling
- RAG
- Vision
- Function calling
- Wake word
