"""
Audio utilities for TTS and STT (PyAudio-free version)
Uses sounddevice for audio input and gTTS for text-to-speech output.
All processing messages are printed to terminal, not Streamlit.
Optimized for use inside Docker containers.
"""

import sounddevice as sd
import numpy as np
import speech_recognition as sr
import threading
import io
from gtts import gTTS
import soundfile as sf


class AudioManager:
    def __init__(self):
        # Initialize Recognizer (Speech-to-Text)
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 2.0
            self.recognizer.non_speaking_duration = 1.5
            self.recognizer.phrase_threshold = 0.3
        except Exception:
            self.recognizer = None

    # ---------------------- TTS ----------------------
    def speak(self, text):
        """Convert text to speech using gTTS and play via sounddevice."""
        if not text:
            return

        try:
            tts = gTTS(text=text, lang='en')
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            # Read MP3 as waveform (float32)
            data, samplerate = sf.read(audio_buffer, dtype='float32')
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")

    def speak_async(self, text):
        """Speak in a background thread (non-blocking)."""
        thread = threading.Thread(target=self.speak, args=(text,))
        thread.daemon = True
        thread.start()

    # ---------------------- STT ----------------------
    def _record_audio(self, duration=15, samplerate=16000):
        """Record audio using sounddevice and return numpy array (max 15s)."""
        try:
            audio_data = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=1,
                dtype='int16'
            )
            sd.wait()
            return audio_data, samplerate
        except Exception as e:
            print(f"⚠️ Microphone error: {e}")
            return None, None

    def _to_audio_data(self, audio_np, samplerate):
        """Convert numpy array to speech_recognition.AudioData."""
        if audio_np is None:
            return None
        audio_bytes = audio_np.tobytes()
        return sr.AudioData(audio_bytes, samplerate, 2)

    def listen(self, timeout=15, phrase_time_limit=15):
        """Listen to user voice input (STT) for up to 15 seconds."""
        if not self.recognizer:
            print("❌ Speech recognizer not initialized.")
            return None

        audio_np, sr_rate = self._record_audio(duration=15)
        if audio_np is None:
            return None

        audio = self._to_audio_data(audio_np, sr_rate)
        try:
            text = self.recognizer.recognize_google(audio)
            print(f" You said: {text}")
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None

    def listen_long_form(self, timeout=15, phrase_time_limit=15):
        """Capture longer responses (max 15 seconds)."""
        if not self.recognizer:
            return None

        audio_np, sr_rate = self._record_audio(duration=15)
        if audio_np is None:
            return None

        audio = self._to_audio_data(audio_np, sr_rate)
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except Exception as e:
            return None

    def test_microphone(self):
        """Quick check if microphone is working."""
        try:
            duration = 2
            audio_data = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
            sd.wait()
            print("✅ Microphone test completed successfully.")
            return True
        except Exception as e:
            print(f"⚠️ Microphone test failed: {e}")
            return False


# ---------------------- Global Helper ----------------------
_audio_manager = None


def get_audio_manager():
    """Get or create a single AudioManager instance."""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = AudioManager()
        _audio_manager.test_microphone()
    return _audio_manager
