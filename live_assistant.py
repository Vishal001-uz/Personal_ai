import sys
import time
import queue
import threading
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
import ollama
from faster_whisper import WhisperModel
import openwakeword
from openwakeword.model import Model

# --- Configuration ---
OLLAMA_MODEL = "qwen2.5:7b" 
PIPER_MODEL = "en_US-amy-medium.onnx"
WHISPER_MODEL = "tiny.en"
WAKE_WORD = "hey_jarvis" 

# --- Global State ---
interrupt_flag = threading.Event()

print("Loading Models (this takes a few seconds on first boot)...")
stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
openwakeword.utils.download_models()
oww_model = Model(wakeword_models=[WAKE_WORD])
print("Models loaded successfully!\n")

def check_for_interruption(threshold=0.03):
    with sd.InputStream(samplerate=16000, channels=1, dtype="int16") as stream:
        while not interrupt_flag.is_set():
            chunk, _ = stream.read(1600)
            rms = np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
            if rms > threshold:
                print("\n[Interrupted!]")
                interrupt_flag.set()
                sd.stop()
                break

def ask_ollama(history, user_text):
    history.append({"role": "user", "content": user_text})
    system_prompt = "You are a helpful, friendly desktop voice assistant. Keep answers highly concise, conversational, and under 3 sentences."
    
    interrupt_flag.clear()
    play_queue = queue.Queue()
    
    def play_audio():
        while True:
            if interrupt_flag.is_set():
                break
            sentence = play_queue.get()
            if sentence is None:
                play_queue.task_done()
                break
                
            subprocess.run([
                "piper", "--model", PIPER_MODEL, "--output_file", "chunk.wav"
            ], input=sentence.encode('utf-8'), stderr=subprocess.DEVNULL)
            
            if not interrupt_flag.is_set():
                data, fs = sf.read("chunk.wav", dtype='float32')
                sd.play(data, fs)
                sd.wait() 
            play_queue.task_done()
            
    player_thread = threading.Thread(target=play_audio)
    player_thread.start()
    
    interrupt_thread = threading.Thread(target=check_for_interruption)
    interrupt_thread.start()
    
    print("Assistant: ", end="", flush=True)
    response_stream = ollama.chat(
        model=OLLAMA_MODEL, 
        messages=[{"role": "system", "content": system_prompt}] + history,
        stream=True
    )
    
    full_text = ""
    buffer = ""
    
    for chunk in response_stream:
        if interrupt_flag.is_set():
            break
            
        token = chunk['message']['content']
        print(token, end="", flush=True)
        full_text += token
        buffer += token
        
        if any(p in token for p in ['.', '?', '!']):
            if buffer.strip():
                play_queue.put(buffer.strip())
                buffer = ""
                
    if buffer.strip() and not interrupt_flag.is_set():
        play_queue.put(buffer.strip())
        
    play_queue.put(None) 
    player_thread.join()
    
    interrupt_flag.set() 
    interrupt_thread.join()
    print("\n") 
    
    history.append({"role": "assistant", "content": full_text})
    return full_text

def wait_for_wake_word():
    print(f"Listening for wake word: '{WAKE_WORD.replace('_', ' ').title()}'...")
    with sd.InputStream(samplerate=16000, channels=1, dtype="int16") as stream:
        while True:
            chunk, _ = stream.read(1280)
            prediction = oww_model.predict(chunk.flatten())
            if max(prediction.values()) > 0.5:
                print("Wake word detected!")
                sys.stdout.write('\a')
                sys.stdout.flush()
                return

def listen_for_command(threshold=0.015, max_seconds=10):
    print("Listening for command... (Speak now)")
    frames = []
    silence_frames = 0
    sample_rate = 16000
    chunk_size = 1600 
    
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            rms = np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
            if rms > threshold:
                frames.append(chunk)
                break
        
        while silence_frames < 15 and len(frames) * chunk_size < sample_rate * max_seconds:
            chunk, _ = stream.read(chunk_size)
            frames.append(chunk)
            rms = np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
            
            if rms < threshold:
                silence_frames += 1
            else:
                silence_frames = 0 
                
    return np.concatenate(frames, axis=0)

def main():
    print(f"--- Local AI Assistant Booted ---")
    conversation_history = []
    
    while True:
        try:
            wait_for_wake_word()
            
            audio_pcm = listen_for_command()
            audio_float = audio_pcm.astype(np.float32) / 32768.0
            
            audio_float = audio_float.reshape(-1)
            segments, _ = stt.transcribe(audio_float, language="en", vad_filter=True)
            user_text = " ".join(s.text.strip() for s in segments)
            
            if not user_text.strip():
                continue
                
            print(f"You: {user_text}")
            
            if "goodbye" in user_text.lower() or "exit" in user_text.lower():
                print("Shutting down...")
                break
                
            ask_ollama(conversation_history, user_text)
            
        except KeyboardInterrupt:
            print("\nForce quitting assistant.")
            break

if __name__ == "__main__":
    main()