"""Transcribe the 60s middle clip and write the result to a UTF-8 file."""
from pathlib import Path
from faster_whisper import WhisperModel

CLIP = Path(r"C:\Users\aayus\Desktop\Meeting_rec\meeting_20260511_115156\middle_60s.wav")
OUT = CLIP.with_name("middle_60s_transcript.txt")

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(
    str(CLIP), language="hi", beam_size=1, vad_filter=True
)

lines = [f"language={info.language} prob={info.language_probability:.2f}\n"]
for seg in segments:
    lines.append(f"[{seg.start:6.1f}s -> {seg.end:6.1f}s]  {seg.text.strip()}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}  ({len(lines)-1} segments)")
