"""
W3 — 全流式

- ASR: 流式
- LLM: 流式 (stream=True, 逐token返回)
- TTS: 流式 (逐句合成, 边生成边播)
"""

import time
import base64
import asyncio
from .base import BasePipeline, PipelineResult, TimingMetrics
from streaming.asr_azure import StreamingASR
from streaming.llm_api import StreamingLLM
from streaming.tts_azure import StreamingTTS
from sentence_splitter import SentenceSplitter


class W3FullStreaming(BasePipeline):
    name = "w3"
    description = "全流式 — 流式ASR → 流式LLM → 句子切分 → 流式TTS"

    def __init__(self, asr: StreamingASR, llm: StreamingLLM, tts: StreamingTTS):
        self.asr = asr
        self.llm = llm
        self.tts = tts

    async def run(self, audio_bytes: bytes) -> PipelineResult:
        t = TimingMetrics()
        t_start = time.perf_counter()

        async def single_chunk():
            yield audio_bytes

        asr_text = ""
        async for etype, text in self.asr.transcribe_stream(single_chunk()):
            if etype == "final":
                asr_text = text

        t.asr = time.perf_counter() - t_start
        if not asr_text.strip():
            return PipelineResult(error="未识别到语音", timings=t)

        full_answer = ""
        splitter = SentenceSplitter()
        tts_audio = b""
        t_llm_start = time.perf_counter()
        first_token = True

        async for etype, text in self.llm.generate_stream(asr_text):
            if etype == "token":
                if first_token:
                    t.llm_ttfb = time.perf_counter() - t_llm_start
                    t.first_audio = t.asr + t.llm_ttfb
                    first_token = False
                full_answer += text
                # Send complete sentences to TTS (non-streaming for experiment)
                sentences = splitter.feed(text)
                for sent in sentences:
                    chunk = await self.tts.synthesize_once(sent)
                    tts_audio += chunk

        # Flush remaining
        for sent in splitter.flush():
            chunk = await self.tts.synthesize_once(sent)
            tts_audio += chunk

        t.llm = time.perf_counter() - t_llm_start
        t.tts = 0  # TTS was interleaved with LLM
        t.total = time.perf_counter() - t_start

        return PipelineResult(
            asr_text=asr_text, answer_text=full_answer,
            tts_audio=tts_audio, timings=t,
        )

    async def run_stream(self, audio_chunk_iter, ws):
        """Full streaming WebSocket handler."""
        t = TimingMetrics()
        t_start = time.perf_counter()
        asr_text = ""

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

        # Streaming LLM → sentence splitter → streaming TTS → send audio
        splitter = SentenceSplitter()
        full_answer = ""
        llm_task = None

        async def _llm_producer():
            nonlocal full_answer
            async for etype, text in self.llm.generate_stream(asr_text):
                if etype == "token":
                    full_answer += text
                    await ws.send_json({"type": "llm_token", "text": text})
                    sentences = splitter.feed(text)
                    for sent in sentences:
                        yield sent
                elif etype == "final":
                    full_answer = text
            for sent in splitter.flush():
                yield sent

        # Stream LLM output and send sentences to TTS in real-time
        t_llm_start = time.perf_counter()
        first_token = True
        async for sentence in _llm_producer():
            if first_token:
                t.llm_ttfb = time.perf_counter() - t_llm_start
                first_token = False
            # Synthesize each sentence and send audio chunk immediately
            audio_chunk = await self.tts.synthesize_once(sentence)
            audio_b64 = base64.b64encode(audio_chunk).decode()
            await ws.send_json({"type": "tts_chunk", "data": audio_b64, "text": sentence})

        t.llm = time.perf_counter() - t_llm_start
        t.total = time.perf_counter() - t_start
        await ws.send_json({"type": "done",
                            "timings": {"asr": round(t.asr, 2), "llm": round(t.llm, 2),
                                        "total": round(t.total, 2)}})

        return PipelineResult(asr_text=asr_text, answer_text=full_answer, timings=t)
