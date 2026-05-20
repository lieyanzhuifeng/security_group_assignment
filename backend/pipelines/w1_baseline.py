"""
W1 — Baseline (非流式)

Non-streaming pipeline using online APIs sequentially.
- ASR: Azure Speech recognize_once
- LLM: DeepSeek generate_once
- TTS: Azure TTS synthesize_once
"""

import time
import base64
from .base import BasePipeline, PipelineResult, TimingMetrics
from streaming.asr_azure import StreamingASR
from streaming.llm_api import StreamingLLM
from streaming.tts_azure import StreamingTTS


class W1Baseline(BasePipeline):
    name = "w1"
    description = "基线 — 非流式: 完整音频 → 完整ASR → 完整LLM → 完整TTS"

    def __init__(self, asr: StreamingASR, llm: StreamingLLM, tts: StreamingTTS):
        self.asr = asr
        self.llm = llm
        self.tts = tts

    async def run(self, audio_bytes: bytes) -> PipelineResult:
        t = TimingMetrics()
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        asr_text = await self.asr.transcribe_once(audio_bytes)
        t.asr = time.perf_counter() - t0
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
        """W1 doesn't support streaming; accumulate and use run()."""
        chunks = []
        async for chunk, is_last in audio_chunk_iter:
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)
        return await self.run(audio_bytes)
