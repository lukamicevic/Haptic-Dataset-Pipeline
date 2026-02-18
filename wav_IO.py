import numpy as np
import wave


def load_wav(filepath: str) -> tuple[np.ndarray, int]:
    with wave.open(filepath, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_array, sample_rate


def save_wav(filepath: str, audio: np.ndarray, sample_rate: int) -> None:
    with wave.open(filepath, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.astype(np.int16).tobytes())
