"""
Streaming TTS using Azure Speech Services.
Uses PushAudioOutputStreamCallback to capture audio chunks as they are synthesized.
"""

import asyncio
import threading
from azure.cognitiveservices import speech as speechsdk


class _AudioChunkCollector(speechsdk.audio.PushAudioOutputStreamCallback):
    """Collects audio chunks from Azure TTS synthesis."""

    def __init__(self):
        super().__init__()
        self.chunks = []
        self._event = threading.Event()

    def write(self, data: bytes) -> int:
        self.chunks.append(data)
        return len(data)

    def close(self):
        self._event.set()

    def wait(self, timeout: float = 30.0):
        self._event.wait(timeout)


class StreamingTTS:
    """Azure TTS with streaming audio output."""

    def __init__(self, key: str, region: str, voice: str = "zh-CN-XiaoxiaoNeural"):
        self._config = speechsdk.SpeechConfig(subscription=key, region=region)
        self._voice = voice

    async def synthesize_stream(self, text: str):
        """
        Synthesize text to speech, yield audio chunks (bytes) as they are generated.
        """
        collector = _AudioChunkCollector()
        push_stream = speechsdk.audio.PushAudioOutputStream(collector)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._config, audio_config=audio_config
        )

        ssml = self._build_ssml(text)
        # Start synthesis (non-blocking)
        synthesizer.speak_ssml_async(ssml)

        # Run the blocking wait in a thread to not block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, collector.wait)

        for chunk in collector.chunks:
            yield chunk

    async def synthesize_once(self, text: str) -> bytes:
        """Non-streaming: return full audio bytes."""
        collector = _AudioChunkCollector()
        push_stream = speechsdk.audio.PushAudioOutputStream(collector)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._config, audio_config=audio_config
        )

        ssml = self._build_ssml(text)
        synthesizer.speak_ssml_async(ssml)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, collector.wait)
        return b"".join(collector.chunks)

    def _build_ssml(self, text: str) -> str:
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="zh-CN">'
            f'<voice name="{self._voice}">'
            f'{text}'
            f'</voice></speak>'
        )
