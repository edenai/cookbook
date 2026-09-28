"""Create samples/question.mp3 (the "Use sample" button) with Eden AI text-to-speech. Costs a fraction of a cent."""
import asyncio
import sys
from pathlib import Path

import edenai

QUESTION = " ".join(sys.argv[1:]) or "What are the biggest AI announcements this week?"


async def main():
    body = await edenai.speak(QUESTION, "audio/tts/gradium")
    audio = edenai.httpx.get(body["output"]["audio_resource_url"])  # plain GET: don't send the key to the file host
    Path("samples").mkdir(exist_ok=True)
    Path("samples/question.mp3").write_bytes(audio.content)
    print(f'saved samples/question.mp3: "{QUESTION}" (cost ${body["cost"]})')

asyncio.run(main())
