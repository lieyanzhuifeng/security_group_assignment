from streaming.asr_azure import StreamingASR
from streaming.llm_api import StreamingLLM
from streaming.tts_azure import StreamingTTS
from .w1_baseline import W1Baseline
from .w2_streaming_asr import W2StreamingASR
from .w3_full_stream import W3FullStreaming
from .w4_full_plus import W4FullPlus

_pipelines = {}


def init_pipelines(asr: StreamingASR, llm: StreamingLLM, tts: StreamingTTS):
    global _pipelines
    _pipelines = {
        "w1": W1Baseline(asr, llm, tts),
        "w2": W2StreamingASR(asr, llm, tts),
        "w3": W3FullStreaming(asr, llm, tts),
        "w4": W4FullPlus(asr, llm, tts),
    }


def get_pipeline(name: str):
    return _pipelines.get(name)


def list_pipelines():
    return {k: v.description for k, v in _pipelines.items()}
