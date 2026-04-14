"""Summarizer module - generates meeting notes via Ollama local LLM."""

import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:26b"

PROMPT_TEMPLATE = """You are an expert meeting note-taker who works in a Hinglish (Hindi + English mix) office environment.

Below is a transcript of a meeting. Convert it into structured meeting notes.

## FORMAT:
### Title
(Give a short, descriptive meeting title)

### Summary
(2-3 line summary in natural Hinglish - mix of Hindi and English as people actually speak)

### Key Discussion Points
- Point 1
- Point 2
- ...

### Action Items
- [ ] Action item 1 (Responsible: person name if mentioned)
- [ ] Action item 2
- ...

### Important Decisions
- Decision 1
- Decision 2
- ...

## INSTRUCTIONS:
- Keep the language in natural Hinglish - the way people actually speak in Indian offices
- Do NOT translate everything to pure Hindi or pure English
- Keep it readable, practical, and concise
- If a person's name is mentioned, attribute action items to them
- If the transcript is short or unclear, do your best with available information

## TRANSCRIPT:
{transcript}

## MEETING NOTES:
"""


class Summarizer:
    """Generates structured meeting notes from transcript using Ollama."""

    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url

    def check_ollama(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                logger.info("Ollama available. Models: %s", models)
                return True
        except Exception:
            logger.warning("Ollama server not reachable at localhost:11434")
            return False

    def list_models(self) -> list[str]:
        """Return list of available Ollama model names."""
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def generate_notes(self, transcript: str, on_token=None) -> str:
        """
        Send transcript to Ollama using streaming and return structured meeting notes.

        Streaming reads tokens one at a time, so there is no single HTTP timeout
        that can kill a slow 26B model mid-generation.

        Args:
            transcript: The full meeting transcript text.
            on_token: Optional callback(str) invoked with each token as it arrives.

        Returns:
            Formatted meeting notes string.

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If the LLM returns an error.
        """
        if not transcript or not transcript.strip():
            return "No transcript available - audio may have been empty or silent."

        prompt = PROMPT_TEMPLATE.format(transcript=transcript)
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
            },
        }).encode("utf-8")

        logger.info("Sending transcript (%d chars) to Ollama model '%s' (streaming)...",
                     len(transcript), self.model)

        try:
            import http.client

            # No timeout at all — gemma4:26b can take minutes to load into memory
            # before it sends the first response byte
            conn = http.client.HTTPConnection("localhost", 11434)
            conn.request("POST", "/api/generate", body=payload,
                         headers={"Content-Type": "application/json"})

            # Wait indefinitely for model to load + first response
            conn.sock.settimeout(None)
            resp = conn.getresponse()

            chunks = []
            for line in resp:
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                token = data.get("response", "")
                if token:
                    chunks.append(token)
                    if on_token:
                        on_token(token)

                if data.get("done", False):
                    break

            conn.close()
            notes = "".join(chunks).strip()
            logger.info("Notes generated: %d characters", len(notes))
            return notes

        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(f"Cannot reach Ollama at {self.ollama_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error generating notes: {e}") from e
