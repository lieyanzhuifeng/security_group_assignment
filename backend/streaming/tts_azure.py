"""
TTS using Azure Speech Services (synchronous API).
"""

import asyncio
import logging
from xml.sax.saxutils import escape as xml_escape
from azure.cognitiveservices import speech as speechsdk

logger = logging.getLogger(__name__)


class StreamingTTS:
    """Azure TTS with streaming audio output."""

    def __init__(self, key: str, region: str, voice: str = "zh-CN-XiaoxiaoNeural"):
        self._config = speechsdk.SpeechConfig(subscription=key, region=region)
        self._voice = voice

    async def synthesize_stream(self, text: str):
        """Synthesize text to speech, yield audio chunks (bytes)."""
        audio_bytes = await self.synthesize_once(text)
        if audio_bytes:
            yield audio_bytes

    async def synthesize_once(self, text: str) -> bytes:
        """Non-streaming: return full audio bytes."""
        ssml = self._build_ssml(text)

        def _synth():
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self._config)
            result = synthesizer.speak_ssml(ssml)
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            logger.error(
                f"TTS synthesis failed: reason={result.reason} "
                f"details={result.error_details}"
            )
            return b""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _synth)

    def _build_ssml(self, text: str) -> str:
        safe_text = xml_escape(text)
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="zh-CN">'
            f'<voice name="{self._voice}">'
            f'{safe_text}'
            f'</voice></speak>'
        )
