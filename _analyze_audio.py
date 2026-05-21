"""One-off diagnostic: cut a 1-minute slice from the middle of a WAV and analyze it."""
import sys
import wave
from pathlib import Path
import numpy as np

SRC = Path(r"C:\Users\aayus\Desktop\Meeting_rec\meeting_20260511_115156\meeting.wav")
DST = SRC.with_name("middle_60s.wav")

with wave.open(str(SRC), "rb") as wf:
    sr = wf.getframerate()
    nch = wf.getnchannels()
    sw = wf.getsampwidth()
    n = wf.getnframes()
    raw = wf.readframes(n)

duration = n / sr
print(f"Source: {SRC.name}")
print(f"  rate={sr} Hz  channels={nch}  sampwidth={sw}B  frames={n}  duration={duration:.1f}s")

dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
if nch > 1:
    audio = audio.reshape(-1, nch).mean(axis=1)
audio /= np.iinfo(dtype).max

# Cut middle 60 seconds
mid = len(audio) // 2
half = sr * 30
start = max(0, mid - half)
end = min(len(audio), mid + half)
clip = audio[start:end]
print(f"\nMiddle slice: samples {start}..{end}  ({(end-start)/sr:.1f}s, "
      f"{start/sr:.1f}s..{end/sr:.1f}s of source)")

# Save the clip
clip_i16 = np.clip(clip * 32767, -32768, 32767).astype(np.int16)
with wave.open(str(DST), "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes(clip_i16.tobytes())
print(f"Saved clip -> {DST}")

# --- Quality metrics ---
print("\n=== Quality metrics on middle 60s ===")
peak = float(np.abs(clip).max())
rms = float(np.sqrt(np.mean(clip ** 2)))
p99 = float(np.quantile(np.abs(clip), 0.99))
p999 = float(np.quantile(np.abs(clip), 0.999))
print(f"  peak={peak:.4f}  rms={rms:.4f}  p99={p99:.4f}  p999={p999:.4f}")

# Clipping samples (within 0.5dB of int16 ceiling)
clip_thresh = 32700 / 32767
clipped = int(np.sum(np.abs(clip) >= clip_thresh))
print(f"  near-clip samples (|x| >= {clip_thresh:.4f}): {clipped} ({100*clipped/len(clip):.3f}%)")

# Dynamic range
dr_db = 20 * np.log10(peak / (rms + 1e-9))
print(f"  crest factor (peak/rms): {dr_db:.1f} dB  (speech is typically 12-18 dB; "
      f">25 dB = lots of silence; <10 dB = compressed/clipped)")

# Silence
window = sr // 10  # 100ms windows
windows = clip[: (len(clip) // window) * window].reshape(-1, window)
win_rms = np.sqrt(np.mean(windows ** 2, axis=1))
silent_frac = float(np.mean(win_rms < 0.01))
loud_frac = float(np.mean(win_rms > 0.3))
print(f"  silent windows (rms<0.01): {100*silent_frac:.1f}%")
print(f"  loud windows  (rms>0.30):  {100*loud_frac:.1f}%")

# Crude DC offset
dc = float(np.mean(clip))
print(f"  DC offset: {dc:+.5f}")

# Spectral centroid (rough timbre indicator)
N = 1 << int(np.ceil(np.log2(min(len(clip), sr * 4))))
spec = np.abs(np.fft.rfft(clip[:N] * np.hanning(N)))
freqs = np.fft.rfftfreq(N, d=1/sr)
centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-9))
print(f"  spectral centroid: {centroid:.0f} Hz  (speech ~500-2000 Hz; "
      f">3000 Hz = hissy/noisy)")

# --- Transcribe the clip ---
print("\n=== Transcribing middle 60s with faster-whisper (small, hi) ===")
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(DST), language="hi", beam_size=1, vad_filter=True)
    print(f"  detected language: {info.language} (prob={info.language_probability:.2f})")
    total = []
    for seg in segments:
        line = f"[{seg.start:6.1f}s -> {seg.end:6.1f}s]  {seg.text.strip()}"
        print("  " + line)
        total.append(seg.text)
    full = " ".join(total).strip()
    print(f"\n  total chars: {len(full)}")
except Exception as e:
    print(f"  transcribe failed: {e}")
