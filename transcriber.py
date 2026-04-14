"""Transcription module - converts audio to text using faster-whisper.

Uses CTranslate2 backend for 4x speed over regular whisper.
Loads WAV files directly via the wave module — no FFmpeg required.
Uses VAD filter to skip silence and avoid hallucination.
Supports progress callback for live UI updates.
"""

import logging
import wave
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "large-v3"
WHISPER_SAMPLE_RATE = 16000


def load_wav_as_float(audio_path: str) -> np.ndarray:
    """
    Read a WAV file and return a float32 numpy array normalized to [-1, 1].
    Resamples to 16kHz mono if needed. No FFmpeg required.
    """
    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if framerate != WHISPER_SAMPLE_RATE:
        duration = len(audio) / framerate
        target_len = int(duration * WHISPER_SAMPLE_RATE)
        indices = np.linspace(0, len(audio) - 1, target_len).astype(int)
        audio = audio[indices]
        logger.info("Resampled from %d Hz to %d Hz", framerate, WHISPER_SAMPLE_RATE)

    return audio


def get_wav_duration(audio_path: str) -> float:
    """Return duration of a WAV file in seconds without loading full audio."""
    with wave.open(audio_path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


class Transcriber:
    """Transcribes audio files using faster-whisper (CTranslate2 backend)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def load_model(self):
        """Load the faster-whisper model. First run downloads the model."""
        if self._model is None:
            logger.info("Loading faster-whisper model '%s'...", self.model_name)
            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",  # Fast on CPU, low memory
            )
            logger.info("faster-whisper model '%s' loaded successfully", self.model_name)

    def transcribe(self, audio_path: str, language: str = "hi",
                   on_progress=None, on_segment=None) -> str:
        """
        Transcribe an audio file and return the full transcript text.

        Args:
            audio_path: Path to the WAV file.
            language: Language hint. 'hi' for Hindi/Hinglish.
            on_progress: Optional callback(percent: float, time_done: float, total: float)
                         called after each segment with transcription progress.
            on_segment: Optional callback(text: str, start: float, end: float)
                        called with each transcribed segment for live display.

        Returns:
            Transcript string. Empty string if no speech detected.
        """
        self.load_model()
        logger.info("Transcribing '%s' (language=%s, model=%s)...",
                     audio_path, language, self.model_name)

        audio = load_wav_as_float(audio_path)
        total_duration = len(audio) / WHISPER_SAMPLE_RATE
        logger.info("Audio loaded: %.1f seconds, %d samples", total_duration, len(audio))

        segments_iter, info = self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300,
            ),
        )

        logger.info("Detected language: %s (prob: %.2f)", info.language, info.language_probability)

        parts = []
        for segment in segments_iter:
            text = segment.text.strip()
            if text:
                parts.append(text)
                logger.info("  [%.1fs -> %.1fs] %s", segment.start, segment.end, text[:80])

                # Report progress based on how far into the audio we are
                if on_segment:
                    on_segment(text, segment.start, segment.end)

                if on_progress and total_duration > 0:
                    percent = min((segment.end / total_duration) * 100, 99.9)
                    on_progress(percent, segment.end, total_duration)

        # Final 100%
        if on_progress:
            on_progress(100.0, total_duration, total_duration)

        transcript = " ".join(parts)

        if not transcript:
            logger.warning("No speech detected in audio")
            return ""

        logger.info("Transcription complete: %d characters", len(transcript))
        return transcript

    def is_silent(self, audio_path: str, threshold: float = 0.005) -> bool:
        """Quick check whether an audio file is mostly silence."""
        try:
            audio = load_wav_as_float(audio_path)
            rms = np.sqrt(np.mean(audio ** 2))
            logger.info("Audio RMS: %.4f (threshold: %.4f)", rms, threshold)
            return rms < threshold
        except Exception:
            logger.exception("Error checking silence")
            return False
