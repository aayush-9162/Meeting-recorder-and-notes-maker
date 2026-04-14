"""Tkinter GUI for Meeting Recording & AI Notes."""

import logging
import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog

from recorder import AudioRecorder
from transcriber import Transcriber, get_wav_duration
from summarizer import Summarizer
from utils import save_text, generate_output_dir, format_duration

logger = logging.getLogger(__name__)

# UI theme colors
BG_COLOR = "#1e1e2e"
FG_COLOR = "#cdd6f4"
ACCENT = "#89b4fa"
RED = "#f38ba8"
GREEN = "#a6e3a1"
YELLOW = "#f9e2af"
SURFACE = "#313244"


class MeetingApp:
    """Main application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Meeting Recorder & AI Notes")
        self.root.geometry("820x720")
        self.root.minsize(720, 620)
        self.root.configure(bg=BG_COLOR)

        # Core components
        self.recorder = AudioRecorder()
        self.transcriber = Transcriber()
        self.summarizer = Summarizer()

        # State
        self._timer_id = None
        self._output_dir = ""
        self._last_audio_path = ""

        self._build_ui()
        self._populate_devices()

    # ── UI Construction ──────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=ACCENT)
        style.configure("Status.TLabel", font=("Segoe UI", 11), foreground=GREEN)
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("TCombobox", font=("Segoe UI", 10))

        # Custom progress bar styles
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor=SURFACE, background=GREEN, thickness=22)
        style.configure("level.Horizontal.TProgressbar",
                        troughcolor=SURFACE, background=ACCENT, thickness=12)

        # Title bar
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=20, pady=(15, 5))
        ttk.Label(title_frame, text="Meeting Recorder & AI Notes", style="Title.TLabel").pack(anchor=tk.W)

        # Device selection
        device_frame = ttk.Frame(self.root)
        device_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(device_frame, text="Input Device:").pack(side=tk.LEFT, padx=(0, 8))
        self.device_combo = ttk.Combobox(device_frame, state="readonly", width=55)
        self.device_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.refresh_btn = ttk.Button(device_frame, text="Refresh", command=self._populate_devices)
        self.refresh_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Gain slider + audio level meter
        gain_frame = ttk.Frame(self.root)
        gain_frame.pack(fill=tk.X, padx=20, pady=(2, 5))
        ttk.Label(gain_frame, text="Mic Gain:").pack(side=tk.LEFT, padx=(0, 5))
        self.gain_var = tk.DoubleVar(value=3.0)
        self.gain_slider = ttk.Scale(gain_frame, from_=1.0, to=10.0,
                                     variable=self.gain_var, orient=tk.HORIZONTAL, length=120)
        self.gain_slider.pack(side=tk.LEFT, padx=(0, 5))
        self.gain_label = ttk.Label(gain_frame, text="3.0x", width=5)
        self.gain_label.pack(side=tk.LEFT, padx=(0, 15))
        self.gain_slider.configure(command=self._on_gain_change)

        ttk.Label(gain_frame, text="Level:").pack(side=tk.LEFT, padx=(0, 5))
        self.level_bar = ttk.Progressbar(
            gain_frame, mode="determinate", length=200,
            style="level.Horizontal.TProgressbar",
        )
        self.level_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.level_label = ttk.Label(gain_frame, text="--", width=5, anchor=tk.E)
        self.level_label.pack(side=tk.RIGHT, padx=(5, 0))

        # Recording controls row
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(fill=tk.X, padx=20, pady=10)

        self.start_btn = tk.Button(
            ctrl_frame, text="Start Recording", font=("Segoe UI", 11, "bold"),
            bg=GREEN, fg="#1e1e2e", activebackground="#77d590",
            relief=tk.FLAT, padx=20, pady=8, command=self._on_start,
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = tk.Button(
            ctrl_frame, text="Stop Recording", font=("Segoe UI", 11, "bold"),
            bg=RED, fg="#1e1e2e", activebackground="#e07090",
            relief=tk.FLAT, padx=20, pady=8, command=self._on_stop, state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.retry_btn = tk.Button(
            ctrl_frame, text="Retry", font=("Segoe UI", 10, "bold"),
            bg=YELLOW, fg="#1e1e2e", activebackground="#f0d080",
            relief=tk.FLAT, padx=15, pady=8, command=self._on_retry, state=tk.DISABLED,
        )
        self.retry_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.folder_btn = tk.Button(
            ctrl_frame, text="Open Folder", font=("Segoe UI", 10, "bold"),
            bg=ACCENT, fg="#1e1e2e", activebackground="#6a9ae0",
            relief=tk.FLAT, padx=15, pady=8, command=self._on_open_folder, state=tk.DISABLED,
        )
        self.folder_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.load_btn = tk.Button(
            ctrl_frame, text="Load WAV", font=("Segoe UI", 10, "bold"),
            bg=SURFACE, fg=FG_COLOR, activebackground="#45475a",
            relief=tk.FLAT, padx=15, pady=8, command=self._on_load_wav,
        )
        self.load_btn.pack(side=tk.LEFT)

        # Status & timer row
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=20, pady=(0, 3))
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)
        self.timer_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.timer_var, style="Status.TLabel").pack(side=tk.RIGHT)

        # Progress bar (determinate for transcription, indeterminate for other steps)
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=20, pady=(0, 3))
        self.progress = ttk.Progressbar(
            progress_frame, mode="determinate", length=400,
            style="green.Horizontal.TProgressbar",
        )
        self.progress.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.progress_label = ttk.Label(progress_frame, text="", width=10, anchor=tk.E)
        self.progress_label.pack(side=tk.RIGHT, padx=(8, 0))

        # Text area for notes / live transcript
        text_frame = ttk.Frame(self.root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        self.text_area = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, font=("Consolas", 10),
            bg=SURFACE, fg=FG_COLOR, insertbackground=FG_COLOR,
            relief=tk.FLAT, padx=10, pady=10,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.text_area.insert(tk.END, "Meeting notes will appear here after recording...\n\n"
                              "Buttons:\n"
                              "  Start/Stop  - Record a new meeting\n"
                              "  Retry       - Re-process last recording if something failed\n"
                              "  Open Folder - Open the folder with saved audio/notes\n"
                              "  Load WAV    - Process any existing WAV file\n")
        self.text_area.config(state=tk.DISABLED)

    def _populate_devices(self):
        """Scan and populate the input device dropdown."""
        devices = AudioRecorder.list_input_devices()
        default_idx = AudioRecorder.get_default_input_device()

        names = [f"[{d['index']}] {d['name']}" for d in devices]
        self.device_combo["values"] = names

        for i, d in enumerate(devices):
            if d["index"] == default_idx:
                self.device_combo.current(i)
                break
        else:
            if names:
                self.device_combo.current(0)

        self._devices = devices

    def _get_selected_device_index(self) -> int | None:
        sel = self.device_combo.current()
        if sel >= 0 and sel < len(self._devices):
            return self._devices[sel]["index"]
        return None

    # ── Actions ──────────────────────────────────────────────────

    def _on_gain_change(self, val):
        """Update gain label and apply to active recorder."""
        g = float(val)
        self.gain_label.configure(text=f"{g:.1f}x")
        if self.recorder.is_recording:
            self.recorder.gain = g

    def _on_start(self):
        device_idx = self._get_selected_device_index()
        self._output_dir = generate_output_dir()
        audio_path = os.path.join(self._output_dir, "meeting.wav")

        gain = self.gain_var.get()
        self.recorder = AudioRecorder(output_path=audio_path, gain=gain)
        self.recorder.set_device(device_idx)
        self.recorder.start()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.retry_btn.config(state=tk.DISABLED)
        self.folder_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.device_combo.config(state=tk.DISABLED)
        self._set_status("Recording...", RED)
        self._set_progress(0)
        self._update_timer()
        self._update_level_meter()
        self._set_text("Recording in progress...\n\nWatch the Level meter — if it stays flat, check your input device.\n")
        logger.info("Recording started -> %s (gain=%.1fx)", audio_path, gain)

    def _on_stop(self):
        self.stop_btn.config(state=tk.DISABLED)
        if self._timer_id:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None
        threading.Thread(target=self._pipeline_from_recording, daemon=True).start()

    def _on_retry(self):
        if not self._last_audio_path or not os.path.exists(self._last_audio_path):
            self._set_status("No audio file to retry", RED)
            return
        self.retry_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._pipeline_process_audio,
                         args=(self._last_audio_path,), daemon=True).start()

    def _on_load_wav(self):
        wav_path = filedialog.askopenfilename(
            title="Select a WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not wav_path:
            return
        self._output_dir = os.path.dirname(wav_path)
        self._last_audio_path = wav_path
        self.start_btn.config(state=tk.DISABLED)
        self.retry_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._pipeline_process_audio,
                         args=(wav_path,), daemon=True).start()

    def _on_open_folder(self):
        folder = os.path.abspath(self._output_dir)
        if os.path.isdir(folder):
            subprocess.Popen(["explorer", folder])
        else:
            self._set_status("Folder not found", RED)

    # ── Pipeline ─────────────────────────────────────────────────

    def _pipeline_from_recording(self):
        try:
            self._set_status("Saving audio...", ACCENT)
            audio_path = self.recorder.stop()
            duration = self.recorder.get_duration()
            self._last_audio_path = audio_path
            logger.info("Audio saved: %s (%.1fs)", audio_path, duration)

            self.root.after(0, lambda: self.folder_btn.config(state=tk.NORMAL))

            if duration < 1.0:
                self._set_status("Recording too short", RED)
                self._set_text("Recording was too short (< 1 second). Please try again.\n")
                self._enable_retry_and_reset()
                return

            self._pipeline_process_audio(audio_path)

        except Exception as e:
            logger.exception("Pipeline error (recording phase)")
            self._set_status("Error saving audio!", RED)
            self._set_text(
                f"Error while saving recording:\n{e}\n\n"
                f"Your audio may still be at:\n{self._output_dir}\n"
            )
            self._enable_retry_and_reset()

    def _pipeline_process_audio(self, audio_path: str):
        try:
            abs_audio = os.path.abspath(audio_path)
            tx_path = os.path.join(self._output_dir, "transcript.txt")

            # Reuse existing transcript if available
            transcript = None
            if os.path.exists(tx_path):
                with open(tx_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                if transcript:
                    logger.info("Reusing existing transcript: %s (%d chars)", tx_path, len(transcript))
                    self._set_status("Transcript found — skipped!", GREEN)
                    self._set_progress(100, "100%")
                    self._set_text(
                        f"--- LIVE TRANSCRIPT ---\n\n"
                        f"{transcript}\n\n"
                        f"(Loaded from existing file — {len(transcript)} chars)\n"
                    )
                else:
                    transcript = None

            # Transcribe if needed
            if not transcript:
                try:
                    audio_duration = get_wav_duration(audio_path)
                except Exception:
                    audio_duration = 0

                dur_str = format_duration(audio_duration) if audio_duration else "?"
                self._set_status(f"Transcribing {dur_str} of audio...", ACCENT)
                self._set_progress(0, "0%")
                self._set_text(
                    f"--- LIVE TRANSCRIPT ---\n"
                    f"Audio: {abs_audio}\n"
                    f"Duration: {dur_str}\n"
                    f"Model: faster-whisper large-v3\n\n"
                )

                def _on_progress(percent, time_done, total):
                    """Update progress bar with transcription percentage."""
                    pct_str = f"{percent:.0f}%"
                    done_str = format_duration(time_done)
                    total_str = format_duration(total)
                    self._set_progress(percent, pct_str)
                    self._set_status(
                        f"Transcribing... {pct_str}  ({done_str} / {total_str})",
                        ACCENT,
                    )

                def _on_segment(text, start, end):
                    """Append each transcribed segment to the text area live."""
                    timestamp = f"[{format_duration(start)} -> {format_duration(end)}]"
                    line = f"{timestamp}  {text}\n"
                    def _append():
                        self.text_area.config(state=tk.NORMAL)
                        self.text_area.insert(tk.END, line)
                        self.text_area.see(tk.END)
                        self.text_area.config(state=tk.DISABLED)
                    self.root.after(0, _append)

                transcript = self.transcriber.transcribe(
                    audio_path, language="hi",
                    on_progress=_on_progress,
                    on_segment=_on_segment,
                )

                if not transcript:
                    self._set_status("No speech detected", RED)
                    self._set_progress(0, "")
                    self._set_text(
                        f"No speech detected in:\n{abs_audio}\n\n"
                        "The audio file is saved. You can:\n"
                        "  - Click 'Retry' to try again\n"
                        "  - Click 'Open Folder' to access the WAV file\n"
                    )
                    self._enable_retry_and_reset()
                    return

                save_text(transcript, tx_path)
                logger.info("Transcript saved: %s", tx_path)
                self._set_status("Transcription complete!", GREEN)
                self._set_progress(100, "100%")

            # ── Step 2: Generate notes ───────────────────────────
            self._set_status("Generating notes via Ollama...", ACCENT)
            # Switch progress bar to indeterminate for LLM generation
            self._start_indeterminate()

            if not self.summarizer.check_ollama():
                self._stop_indeterminate()
                self._set_status("Ollama not running — transcript saved", YELLOW)
                self._set_text(
                    f"Transcript saved to:\n{os.path.abspath(tx_path)}\n\n"
                    "Ollama is not running. Start it and click 'Retry':\n"
                    "  ollama serve\n"
                    "  ollama pull gemma4:26b\n\n"
                    f"{'=' * 60}\n"
                    f"--- TRANSCRIPT ---\n\n{transcript}\n"
                )
                self._enable_retry_and_reset()
                return

            models = self.summarizer.list_models()
            if models and not any("gemma" in m for m in models):
                self.summarizer.model = models[0]
                logger.info("gemma not found, using model: %s", models[0])

            # Stream notes live
            self._set_text("--- GENERATING MEETING NOTES ---\n\n")

            def _on_token(token: str):
                def _append():
                    self.text_area.config(state=tk.NORMAL)
                    self.text_area.insert(tk.END, token)
                    self.text_area.see(tk.END)
                    self.text_area.config(state=tk.DISABLED)
                self.root.after(0, _append)

            notes = self.summarizer.generate_notes(transcript, on_token=_on_token)

            # Save notes
            notes_path = os.path.join(self._output_dir, "notes.txt")
            save_text(notes, notes_path)
            logger.info("Notes saved: %s", notes_path)

            # Final display
            output = (
                f"--- MEETING NOTES ---\n\n{notes}\n\n"
                f"{'=' * 60}\n"
                f"--- TRANSCRIPT ---\n\n{transcript}\n\n"
                f"{'=' * 60}\n"
                f"Files saved in: {os.path.abspath(self._output_dir)}\n"
                f"  - meeting.wav\n  - transcript.txt\n  - notes.txt\n"
            )
            self._set_text(output)
            self._stop_indeterminate()
            self._set_progress(100, "Done!")
            self._set_status("Done!", GREEN)
            self._enable_retry_and_reset()

        except Exception as e:
            logger.exception("Pipeline error (processing phase)")
            abs_audio = os.path.abspath(audio_path)
            self._stop_indeterminate()
            self._set_progress(0, "Error")
            self._set_status("Error — audio is safe, click Retry", RED)
            self._set_text(
                f"An error occurred during processing:\n{e}\n\n"
                f"Your audio is safely saved at:\n{abs_audio}\n\n"
                "You can:\n"
                "  - Click 'Retry' to try transcription again\n"
                "  - Click 'Open Folder' to access your files\n"
                "  - Click 'Load WAV' to process a different file\n"
            )
            self._enable_retry_and_reset()

    # ── UI helpers ───────────────────────────────────────────────

    def _set_status(self, text: str, color: str = FG_COLOR):
        self.root.after(0, lambda: (
            self.status_var.set(text),
            self.status_label.configure(foreground=color),
        ))

    def _set_text(self, content: str):
        def _update():
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, content)
            self.text_area.see(tk.END)
            self.text_area.config(state=tk.DISABLED)
        self.root.after(0, _update)

    def _set_progress(self, value: float, label: str = ""):
        """Set determinate progress bar value (0-100) and label text."""
        def _update():
            self.progress.configure(mode="determinate", value=value)
            self.progress_label.configure(text=label)
        self.root.after(0, _update)

    def _start_indeterminate(self):
        """Switch progress bar to indeterminate (bouncing) mode."""
        def _update():
            self.progress.configure(mode="indeterminate")
            self.progress.start(15)
            self.progress_label.configure(text="...")
        self.root.after(0, _update)

    def _stop_indeterminate(self):
        """Stop indeterminate progress and reset."""
        def _update():
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            self.progress_label.configure(text="")
        self.root.after(0, _update)

    def _update_timer(self):
        if self.recorder.is_recording:
            dur = format_duration(self.recorder.get_duration())
            self.timer_var.set(dur)
            self._timer_id = self.root.after(500, self._update_timer)

    def _update_level_meter(self):
        """Update the live audio level meter during recording."""
        if self.recorder.is_recording:
            level = self.recorder.get_current_level()
            # Scale to 0-100 for the progress bar (level is RMS, typically 0-0.5)
            bar_val = min(level * 300, 100)  # Amplify for visibility
            self.level_bar.configure(value=bar_val)
            # Show dB-like label
            if level > 0.001:
                self.level_label.configure(text=f"{bar_val:.0f}%")
            else:
                self.level_label.configure(text="LOW")
            self.root.after(100, self._update_level_meter)
        else:
            self.level_bar.configure(value=0)
            self.level_label.configure(text="--")

    def _enable_retry_and_reset(self):
        def _do():
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.load_btn.config(state=tk.NORMAL)
            self.device_combo.config(state="readonly")
            self.timer_var.set("")
            if self._last_audio_path and os.path.exists(self._last_audio_path):
                self.retry_btn.config(state=tk.NORMAL)
                self.folder_btn.config(state=tk.NORMAL)
        self.root.after(0, _do)
