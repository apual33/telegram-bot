from pathlib import Path
from openai import AsyncOpenAI


async def transcribe(audio_path: Path, client: AsyncOpenAI) -> str:
    """Transcribe an OGG voice file using OpenAI Whisper API."""
    with open(audio_path, "rb") as f:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=("voice.ogg", f, "audio/ogg"),
        )
    return response.text
