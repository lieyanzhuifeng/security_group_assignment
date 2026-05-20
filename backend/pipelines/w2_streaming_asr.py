"""
W2 — 流式ASR

- ASR: 流式 (partial results)
- LLM: 非流式 (等到ASR final后一次生成)
- TTS: 非流式
"""

import time
import base64
import asyncio
from .base import BasePipeline, PipelineResult, TimingMetrics
from streaming.asr_azure import StreamingASR
from streaming.llm_api import StreamingLLM
from streaming.tts_azure import StreamingTTS


class W2StreamingASR(BasePipeline):
    name = "w2"
    description = "流式ASR — 流式语音识别(partial结果) + 非流式LLM/TTS"

    def __init__(self, asr: StreamingASR, llm: StreamingLLM, tts: StreamingTTS):
        self.asr = asr
        self.llm = llm
        self.tts = tts

    async def run(self, audio_bytes: bytes) -> PipelineResult:
        t = TimingMetrics()
        t_start = time.perf_counter()

        # Simulate streaming with a single chunk
        async def single_chunk():
            yield audio_bytes

        asr_text = ""
        async for etype, text in self.asr.transcribe_stream(single_chunk()):
            if etype == "final":
                asr_text = text

        t.asr = time.perf_counter() - t_start
        if not asr_text.strip():
            return PipelineResult(error="未识别到语音", asr_text=asr_text, timings=t)

        t0 = time.perf_counter()
        answer_text = await self.llm.generate_once(asr_text)
        t.llm = time.perf_counter() - t0

        t0 = time.perf_counter()
        tts_audio = await self.tts.synthesize_once(answer_text)
        t.tts = time.perf_counter() - t0

        t.total = time.perf_counter() - t_start
        t.first_audio = t.total

        return PipelineResult(
            asr_text=asr_text, answer_text=answer_text,
            tts_audio=tts_audio, timings=t,
        )

    async def run_stream(self, audio_chunk_iter, ws):
        """
        Full streaming: feed audio chunks -> ASR streaming -> yield partial text.
        On ASR final -> LLM non-streaming -> TTS non-streaming.
        """
        t = TimingMetrics()
        t_start = time.perf_counter()
        asr_text = ""

        # Source: aiter_chunks yields (chunk, is_last)
        async def _audio_gen():
            async for chunk, is_last in audio_chunk_iter:
                yield chunk

        async for etype, text in self.asr.transcribe_stream(_audio_gen()):
            if etype == "partial":
                await ws.send_json({"type": "asr_partial", "text": text})
            elif etype == "final":
                asr_text = text
                await ws.send_json({"type": "asr_final", "text": text})

        t.asr = time.perf_counter() - t_start
        if not asr_text.strip():
            await ws.send_json({"error": "未识别到语音"})
            return PipelineResult(error="未识别到语音", timings=t)

        # LLM non-streaming
        t0 = time.perf_counter()
        answer_text = await self.llm.generate_once(asr_text)
        t.llm = time.perf_counter() - t0
        await ws.send_json({"type": "answer", "text": answer_text})

        # TTS non-streaming
        t0 = time.perf_counter()
        tts_audio = await self.tts.synthesize_once(answer_text)
        t.tts = time.perf_counter() - t0
        audio_b64 = base64.b64encode(tts_audio).decode()
        await ws.send_json({"type": "audio", "data": audio_b64,
                            "timings": {"asr": round(t.asr, 2), "llm": round(t.llm, 2),
                                        "tts": round(t.tts, 2), "total": round(time.perf_counter() - t_start, 2)}})

        t.total = time.perf_counter() - t_start
        return PipelineResult(asr_text=asr_text, answer_text=answer_text,
                              tts_audio=tts_audio, timings=t)
