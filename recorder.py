"""Audio recorder module - captures microphone + system audio via WASAPI loopback."""

import logging
import threading
import wave
import numpy as np
import sounddevice as sd
import soundcard as sc

logger = logging.getLogger(__name__)

RECORD_SAMPLE_RATE = 44100
CHANNELS = 1
BLOCKSIZE = 4096


class AudioRecorder:
    """
    Records microphone input and system audio (WASAPI loopback) simultaneously.
    System audio is captured directly from the speaker output — perfect digital quality.
    Both streams are mixed together into a single WAV file.
    """

    def __init__(self, output_path: str = "meeting.wav", gain: float = 1.0):
        self.output_path = output_path
        self.gain = gain  # Mic gain multiplier
        self.system_gain = 1.0  # System audio gain
        self._recording = False
        self._mic_frames: list[np.ndarray] = []
        self._sys_frames: list[np.ndarray] = []
        self._mic_thread: threading.Thread | None = None
        self._sys_thread: threading.Thread | None = None
        self._mic_device: int | None = None
        self._sample_rate = RECORD_SAMPLE_RATE
        self._peak_level = 0.0
        self._current_level = 0.0
        self._record_system = True  # Enable system audio capture

    # ── Device listing ───────────────────────────────────────────

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
        """Set the microphone input device."""
        self._mic_device = device_index
        if device_index is not None:
            try:
                dev_info = sd.query_devices(device_index)
                native_sr = int(dev_info["default_samplerate"])
                if 22050 <= native_sr <= 192000:
                    self._sample_rate = native_sr
                    logger.info("Using device native sample rate: %d Hz", native_sr)
            except Exception:
                pass
        logger.info("Mic device set to index: %s (rate: %d Hz)", device_index, self._sample_rate)

    # ── Recording ────────────────────────────────────────────────

    def _record_mic_loop(self):
        """Record from microphone using sounddevice."""
        try:
            stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=CHANNELS,
                dtype="float32",
                blocksize=BLOCKSIZE,
                device=self._mic_device,
            )
            stream.start()
            logger.info("Mic recording started (device=%s, rate=%d)", self._mic_device, self._sample_rate)

            while self._recording:
                data, overflowed = stream.read(BLOCKSIZE)
                if overflowed:
                    logger.warning("Mic input overflowed")
                # Apply mic gain
                boosted = data * self.gain
                self._mic_frames.append(boosted.flatten().copy())
                # Update level meter
                level = float(np.sqrt(np.mean(boosted ** 2)))
                self._current_level = level
                peak = float(np.abs(boosted).max())
                if peak > self._peak_level:
                    self._peak_level = peak

            stream.stop()
            stream.close()
            logger.info("Mic recording stopped, %d blocks", len(self._mic_frames))
        except Exception:
            logger.exception("Mic recording error")

    def _record_system_loop(self):
        """Record system audio via WASAPI loopback using soundcard."""
        try:
            # Get loopback device from default speaker
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            logger.info("System audio loopback: %s", loopback)

            with loopback.recorder(samplerate=self._sample_rate, channels=1) as rec:
                logger.info("System audio recording started (loopback, rate=%d)", self._sample_rate)
                while self._recording:
                    # Record 100ms chunks
                    chunk_size = self._sample_rate // 10
                    data = rec.record(numframes=chunk_size)
                    # data shape: (chunk_size, channels) — downmix to mono.
                    # Use 0.5*(L+R) which preserves common content; if L and R are
                    # severely out of phase the energy drops, but for meeting apps
                    # this is the right call (mean would attenuate centered voices
                    # by 6dB only — same here, but explicit).
                    if data.ndim > 1 and data.shape[1] > 1:
                        mono = data.sum(axis=1) * (1.0 / data.shape[1])
                    else:
                        mono = data.flatten()
                    self._sys_frames.append(mono.astype(np.float32) * self.system_gain)

            logger.info("System audio recording stopped, %d blocks", len(self._sys_frames))
        except Exception:
            logger.exception("System audio recording failed — continuing with mic only")
            self._record_system = False

    def start(self):
        """Start recording mic + system audio in background threads."""
        if self._recording:
            logger.warning("Already recording")
            return
        self._mic_frames = []
        self._sys_frames = []
        self._peak_level = 0.0
        self._current_level = 0.0
        self._recording = True

        # Start mic recording thread
        self._mic_thread = threading.Thread(target=self._record_mic_loop, daemon=True)
        self._mic_thread.start()

        # Start system audio recording thread
        if self._record_system:
            self._sys_thread = threading.Thread(target=self._record_system_loop, daemon=True)
            self._sys_thread.start()

    def stop(self) -> str:
        """Stop recording, mix mic + system audio, normalize, and save WAV."""
        if not self._recording:
            logger.warning("Not currently recording")
            return self.output_path

        self._recording = False

        if self._mic_thread:
            self._mic_thread.join(timeout=5)
            self._mic_thread = None
        if self._sys_thread:
            self._sys_thread.join(timeout=5)
            self._sys_thread = None

        # Concatenate mic audio
        mic_audio = np.concatenate(self._mic_frames) if self._mic_frames else np.array([], dtype=np.float32)
        sys_audio = np.concatenate(self._sys_frames) if self._sys_frames else np.array([], dtype=np.float32)

        logger.info("Mic audio: %d samples (%.1fs)", len(mic_audio), len(mic_audio) / self._sample_rate if len(mic_audio) else 0)
        logger.info("System audio: %d samples (%.1fs)", len(sys_audio), len(sys_audio) / self._sample_rate if len(sys_audio) else 0)

        # Mix both streams — match lengths first
        if len(mic_audio) > 0 and len(sys_audio) > 0:
            # Trim to the shorter length so end-padding zeros don't tail the file
            target_len = min(len(mic_audio), len(sys_audio))
            mic_audio = mic_audio[:target_len]
            sys_audio = sys_audio[:target_len]

            # Balance levels: scale each stream to its own peak ~0.9 so a loud
            # remote speaker can't drown out the quieter local mic (and vice versa).
            mic_audio = self._prepare_stream(mic_audio, "mic")
            sys_audio = self._prepare_stream(sys_audio, "system")

            # Weighted mix — 0.5 each so the worst-case sum is exactly 1.0 (no clipping).
            mixed = 0.5 * mic_audio + 0.5 * sys_audio
            logger.info("Mixed mic + system audio: %d samples", len(mixed))
        elif len(mic_audio) > 0:
            mixed = self._prepare_stream(mic_audio, "mic")
            logger.info("Using mic audio only (no system audio)")
        elif len(sys_audio) > 0:
            mixed = self._prepare_stream(sys_audio, "system")
            logger.info("Using system audio only (no mic audio)")
        else:
            logger.warning("No audio captured from any source")
            return self.output_path

        duration = len(mixed) / self._sample_rate

        # Light final normalize so the WAV uses the full int16 range without re-clipping.
        mixed = self._normalize(mixed, target_peak=0.95)

        # Convert to int16 for WAV
        audio_i16 = np.clip(mixed * 32767, -32768, 32767).astype(np.int16)

        logger.info("Saving %.1f seconds to %s (rate=%d Hz)", duration, self.output_path, self._sample_rate)

        with wave.open(self.output_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_i16.tobytes())

        return self.output_path

    @staticmethod
    def _normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """Normalize audio so the loudest peak reaches target_peak."""
        peak = np.abs(audio).max()
        if peak < 1e-6:
            logger.warning("Audio is essentially silent (peak=%.6f)", peak)
            return audio
        scale = target_peak / peak
        scale = min(scale, 20.0)  # Cap gain to avoid amplifying noise
        logger.info("Normalizing: peak=%.4f, scale=%.2fx", peak, scale)
        return audio * scale

    @staticmethod
    def _prepare_stream(audio: np.ndarray, label: str, target_peak: float = 0.9) -> np.ndarray:
        """Per-stream normalize using a robust peak (99.9th percentile) so brief
        clicks/clipping don't crush the rest of the audio. Keeps mic and system
        audio at comparable loudness before they're mixed."""
        if len(audio) == 0:
            return audio
        # 99.9th percentile of |audio| — ignores rare spikes, captures real loudness.
        robust_peak = float(np.quantile(np.abs(audio), 0.999))
        true_peak = float(np.abs(audio).max())
        if robust_peak < 1e-6:
            logger.warning("%s stream is essentially silent (peak=%.6f)", label, true_peak)
            return audio
        scale = target_peak / robust_peak
        scale = min(scale, 10.0)  # cap to avoid blowing up the noise floor
        scaled = audio * scale
        # Soft-limit anything that ended up above 1.0 (brief spikes) instead of hard-clipping.
        scaled = np.tanh(scaled)
        logger.info("%s stream: robust_peak=%.4f true_peak=%.4f scale=%.2fx",
                    label, robust_peak, true_peak, scale)
        return scaled.astype(np.float32)

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def peak_level(self) -> float:
        return self._peak_level

    def get_duration(self) -> float:
        """Return approximate duration of recorded audio so far in seconds."""
        if not self._mic_frames:
            return 0.0
        total = sum(len(f) for f in self._mic_frames)
        return total / self._sample_rate

    def get_current_level(self) -> float:
        """Return the RMS level of the most recent mic audio block."""
        return self._current_level
