"""
Meeting Recorder & AI Notes Generator
======================================
Records meeting audio, transcribes Hindi/Hinglish speech with Whisper,
and generates structured notes using Ollama (local LLM).

Usage:
    python main.py
"""

import tkinter as tk
from utils import setup_logging
from ui import MeetingApp


def main():
    setup_logging()
    root = tk.Tk()
    MeetingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
