"""Audio recorder module - captures microphone and system audio at high quality."""

import logging
import threading
import wave
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Record at high quality — transcriber handles downsampling to 16kHz for Whisper
RECORD_SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = "float32"  # float32 captures full dynamic range, convert to int16 on save
BLOCKSIZE = 4096   # Larger blocks = smoother recording, fewer glitches


class AudioRecorder:
    """Records audio from microphone and/or system audio at high quality."""

    def __init__(self, output_path: str = "meeting.wav", gain: float = 1.0):
        self.output_path = output_path
        self.gain = gain  # Volume multiplier (1.0 = normal, 2.0 = 2x louder)
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._thread: threading.Thread | None = None
        self._mic_device: int | None = None
        self._stream: sd.InputStream | None = None
        self._sample_rate = RECORD_SAMPLE_RATE
        self._peak_level = 0.0  # Track peak audio level during recording

    @staticmethod
    def list_input_devices() -> list[dict]:
        """Return all available input devices with index, name, and default sample rate."""
        devices = sd.query_devices()
        inputs = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                inputs.append({
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "default_sr": d["default_samplerate"],
                })
        return inputs

    @staticmethod
    def get_default_input_device() -> int | None:
        """Return index of the default input device."""
        try:
            info = sd.query_devices(kind="input")
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d["name"] == info["name"]:
                    return i
        except Exception:
            pass
        return None

    def set_device(self, device_index: int | None):
        """Set the input device to record from."""
        self._mic_device = device_index
        # Use device's native sample rate for best quality
        if device_index is not None:
            try:
                dev_info = sd.query_devices(device_index)
                native_sr = int(dev_info["default_samplerate"])
                # Use native rate if it's reasonable (22050-192000)
                if 22050 <= native_sr <= 192000:
                    self._sample_rate = native_sr
                    logger.info("Using device native sample rate: %d Hz", native_sr)
            except Exception:
                pass
        logger.info("Input device set to index: %s (rate: %d Hz)", device_index, self._sample_rate)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback invoked by sounddevice for each audio block."""
        if status:
            logger.warning("Audio stream status: %s", status)
        # Apply gain boost in real-time
        boosted = indata * self.gain
        self._frames.append(boosted.copy())
        # Track peak level for monitoring
        peak = np.abs(boosted).max()
        if peak > self._peak_level:
            self._peak_level = float(peak)

    def _record_loop(self):
        """Internal recording loop running in a separate thread."""
        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCKSIZE,
                device=self._mic_device,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Recording started (device=%s, rate=%d, gain=%.1fx)",
                        self._mic_device, self._sample_rate, self.gain)

            while self._recording:
                sd.sleep(100)

            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Recording stopped, captured %d blocks, peak level: %.4f",
                        len(self._frames), self._peak_level)
        except Exception:
            logger.exception("Error during recording")
            self._recording = False

    def start(self):
        """Start recording audio in a background thread."""
        if self._recording:
            logger.warning("Already recording")
            return
        self._frames = []
        self._peak_level = 0.0
        self._recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self) -> str:
        """Stop recording, normalize volume, save WAV file, and return the file path."""
        if not self._recording:
            logger.warning("Not currently recording")
            return self.output_path

        self._recording = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        if not self._frames:
            logger.warning("No audio frames captured")
            return self.output_path

        # Concatenate all frames (float32, range roughly -1.0 to 1.0)
        audio_f32 = np.concatenate(self._frames, axis=0).flatten()
        duration = len(audio_f32) / self._sample_rate
        logger.info("Raw audio: %d samples, %.1f seconds, peak=%.4f",
                     len(audio_f32), duration, self._peak_level)

        # Normalize volume — boost quiet audio so peak reaches ~90% of max
        audio_f32 = self._normalize(audio_f32)

        # Convert float32 to int16 for WAV
        audio_i16 = np.clip(audio_f32 * 32767, -32768, 32767).astype(np.int16)

        logger.info("Saving %.1f seconds to %s (rate=%d Hz)",
                     duration, self.output_path, self._sample_rate)

        with wave.open(self.output_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_i16.tobytes())

        return self.output_path

    @staticmethod
    def _normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """
        Normalize audio so the loudest peak reaches target_peak.
        Prevents clipping while maximizing volume.
        """
        peak = np.abs(audio).max()
        if peak < 1e-6:
            logger.warning("Audio is essentially silent (peak=%.6f)", peak)
            return audio

        scale = target_peak / peak
        # Cap the gain to avoid amplifying noise too much (max 20x boost)
        scale = min(scale, 20.0)
        logger.info("Normalizing audio: peak=%.4f, scale=%.2fx", peak, scale)
        return audio * scale

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def peak_level(self) -> float:
        """Current peak audio level (0.0 to 1.0+). Use to show level meter."""
        return self._peak_level

    def get_duration(self) -> float:
        """Return approximate duration of recorded audio so far in seconds."""
        if not self._frames:
            return 0.0
        total_samples = sum(f.shape[0] for f in self._frames)
        return total_samples / self._sample_rate

    def get_current_level(self) -> float:
        """Return the RMS level of the most recent audio block (for live level meter)."""
        if not self._frames:
            return 0.0
        last = self._frames[-1]
        return float(np.sqrt(np.mean(last ** 2)))
