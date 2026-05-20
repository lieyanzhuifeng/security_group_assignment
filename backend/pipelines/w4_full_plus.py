"""
W4 — 全量优化

W3 (全流式) + 安全过滤 + 领域热词提示。
- ASR: 流式
- LLM: 流式 + 领域system prompt
- TTS: 流式
- 安全: ASR输出后 → check_safety → 拦截敏感/无关问题
"""

import time
import base64
import asyncio
from .base import BasePipeline, PipelineResult, TimingMetrics
from streaming.asr_azure import StreamingASR
from streaming.llm_api import StreamingLLM
from streaming.tts_azure import StreamingTTS
from sentence_splitter import SentenceSplitter
from safety import check_safety


class W4FullPlus(BasePipeline):
    name = "w4"
    description = "全量优化 — 全流式 + 安全过滤 + 领域热词"

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

        # Safety check
        safety = check_safety(asr_text)
        if not safety["safe"]:
            answer_text = safety["response"]
            t0 = time.perf_counter()
            tts_audio = await self.tts.synthesize_once(answer_text)
            t.tts = time.perf_counter() - t0
            t.total = time.perf_counter() - t_start
            return PipelineResult(
                asr_text=asr_text, answer_text=answer_text, tts_audio=tts_audio,
                timings=t, metrics={"safe": False, "intercept_reason": safety["reason"]},
            )

        # Streaming LLM
        full_answer = ""
        splitter = SentenceSplitter()
        tts_audio = b""
        t_llm_start = time.perf_counter()
        first_token = True

        async for etype, text in self.llm.generate_stream(asr_text):
            if etype == "token":
                if first_token:
                    t.llm_ttfb = time.perf_counter() - t_llm_start
                    first_token = False
                full_answer += text
                sentences = splitter.feed(text)
                for sent in sentences:
                    chunk = await self.tts.synthesize_once(sent)
                    tts_audio += chunk

        for sent in splitter.flush():
            chunk = await self.tts.synthesize_once(sent)
            tts_audio += chunk

        t.llm = time.perf_counter() - t_llm_start
        t.total = time.perf_counter() - t_start

        return PipelineResult(
            asr_text=asr_text, answer_text=full_answer, tts_audio=tts_audio,
            timings=t, metrics={"safe": True},
        )

    async def run_stream(self, audio_chunk_iter, ws):
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

        # Safety check
        safety = check_safety(asr_text)
        if not safety["safe"]:
            answer_text = safety["response"]
            await ws.send_json({"type": "answer", "text": answer_text,
                                "safe": False, "reason": safety["reason"]})
            t0 = time.perf_counter()
            tts_audio = await self.tts.synthesize_once(answer_text)
            t.tts = time.perf_counter() - t0
            audio_b64 = base64.b64encode(tts_audio).decode()
            await ws.send_json({"type": "audio", "data": audio_b64})
            t.total = time.perf_counter() - t_start
            return PipelineResult(asr_text=asr_text, answer_text=answer_text,
                                  tts_audio=tts_audio, timings=t,
                                  metrics={"safe": False, "intercept_reason": safety["reason"]})

        # Streaming LLM → splitter → streaming TTS
        splitter = SentenceSplitter()
        full_answer = ""
        t_llm_start = time.perf_counter()
        first_token = True

        async def _llm_producer():
            nonlocal full_answer
            async for etype, text in self.llm.generate_stream(asr_text):
                if etype == "token":
                    full_answer += text
                    await ws.send_json({"type": "llm_token", "text": text})
                    sentences = splitter.feed(text)
                    for sent in sentences:
                        yield sent
            for sent in splitter.flush():
                yield sent

        async for sentence in _llm_producer():
            if first_token:
                t.llm_ttfb = time.perf_counter() - t_llm_start
                first_token = False
            audio_chunk = await self.tts.synthesize_once(sentence)
            audio_b64 = base64.b64encode(audio_chunk).decode()
            await ws.send_json({"type": "tts_chunk", "data": audio_b64, "text": sentence})

        t.llm = time.perf_counter() - t_llm_start
        t.total = time.perf_counter() - t_start
        await ws.send_json({"type": "done", "safe": True,
                            "timings": {"asr": round(t.asr, 2), "llm": round(t.llm, 2),
                                        "total": round(t.total, 2)}})

        return PipelineResult(asr_text=asr_text, answer_text=full_answer, timings=t,
                              metrics={"safe": True})
