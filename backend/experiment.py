import json
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from streaming.asr_azure import StreamingASR
from streaming.llm_api import StreamingLLM
from streaming.tts_azure import StreamingTTS
from pipelines.factory import init_pipelines, get_pipeline, list_pipelines


def load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_single_test(pname: str, tc: dict) -> dict:
    pipeline = get_pipeline(pname)
    if pipeline is None:
        return {"error": f"unknown: {pname}", "id": tc["id"]}

    audio_path = tc["audio"]
    if not os.path.exists(audio_path):
        return {"error": f"file not found", "id": tc["id"]}

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    result = await pipeline.run(audio_bytes)

    m = {
        "id": tc["id"],
        "error": result.error,
        "asr_time": round(result.timings.asr, 3),
        "llm_time": round(result.timings.llm, 3),
        "tts_time": round(result.timings.tts, 3),
        "total_time": round(result.timings.total, 3),
        "llm_ttfb": round(result.timings.llm_ttfb, 3) if result.timings.llm_ttfb else None,
        "first_audio": round(result.timings.first_audio, 3) if result.timings.first_audio else None,
        "asr_text": result.asr_text,
        "answer_text": result.answer_text,
        "safe": result.metrics.get("safe"),
    }

    # ASR keyword recall
    expected_kws = tc.get("expected_asr_keywords", [])
    if expected_kws:
        hits = sum(1 for kw in expected_kws if kw in result.asr_text)
        m["asr_recall"] = round(hits / len(expected_kws), 2)
    else:
        m["asr_recall"] = None

    # Safety accuracy
    expected_safe = tc.get("safe", True)
    actual_safe = result.metrics.get("safe", True)
    m["safety_correct"] = actual_safe == expected_safe

    return m


async def run_experiment(manifest_path: str):
    test_cases = load_manifest(manifest_path)
    test_dir = os.path.dirname(manifest_path)
    for tc in test_cases:
        if not os.path.isabs(tc["audio"]):
            tc["audio"] = os.path.join(test_dir, tc["audio"])

    pnames = ["w1", "w2", "w3", "w4"]
    pdescs = list_pipelines()
    all_results = {}

    print("=" * 68)
    print("  实验对比 — 船舶建造智能问答系统 (在线API)")
    print("=" * 68)

    for pname in pnames:
        desc = pdescs.get(pname, "")
        print(f"\n{'─' * 68}")
        print(f"  [{pname}] {desc}")
        print(f"{'─' * 68}")
        results = []
        for tc in test_cases:
            if not os.path.exists(tc["audio"]):
                print(f"    ⚠ 跳过 {tc['id']}: 文件不存在")
                continue
            result = await run_single_test(pname, tc)
            results.append(result)
            status = "✓" if not result.get("error") else "✗"
            a = result.get("asr_time","?"); l = result.get("llm_time","?")
            t = result.get("tts_time","?"); tot = result.get("total_time","?")
            r = result.get("asr_recall","N/A")
            print(f"    {status} {result['id']}: ASR={a}s  LLM={l}s  TTS={t}s  total={tot}s  recall={r}")
        all_results[pname] = results

    # Summary table
    print(f"\n\n{'=' * 68}")
    print("  汇总对比")
    print(f"{'=' * 68}")
    hdr = f"{'工作流':<8} {'总延迟':<9} {'首帧延迟':<10} {'ASR':<8} {'LLM':<8} {'TTS':<8} {'LLM-TTFB':<10} {'ASR召回':<8} {'安全':<6}"
    print(f"\n{hdr}\n{'-'*68}")
    for pname in pnames:
        r = all_results.get(pname, [])
        if not r:
            continue
        times = [x["total_time"] for x in r if x.get("total_time")]
        fas = [x["first_audio"] for x in r if x.get("first_audio")]
        asrs = [x["asr_time"] for x in r if x.get("asr_time")]
        llms = [x["llm_time"] for x in r if x.get("llm_time")]
        ttss = [x["tts_time"] for x in r if x.get("tts_time")]
        ttfb = [x["llm_ttfb"] for x in r if x.get("llm_ttfb")]
        recalls = [x["asr_recall"] for x in r if x.get("asr_recall") is not None]
        safe = [x["safety_correct"] for x in r if "safety_correct" in x]
        avg_t = f"{sum(times)/len(times):.2f}s" if times else "N/A"
        avg_f = f"{sum(fas)/len(fas):.2f}s" if fas else "N/A"
        avg_a = f"{sum(asrs)/len(asrs):.2f}s" if asrs else "N/A"
        avg_l = f"{sum(llms)/len(llms):.2f}s" if llms else "N/A"
        avg_tt = f"{sum(ttss)/len(ttss):.2f}s" if ttss else "N/A"
        avg_tf = f"{sum(ttfb)/len(ttfb):.2f}s" if ttfb else "N/A"
        avg_r = f"{sum(recalls)/len(recalls):.2f}" if recalls else "N/A"
        saf = f"{sum(safe)}/{len(safe)}" if safe else "N/A"
        print(f"{pname:<8} {avg_t:<9} {avg_f:<10} {avg_a:<8} {avg_l:<8} {avg_tt:<8} {avg_tf:<10} {avg_r:<8} {saf:<6}")
    print("-" * 68)
    print(f"\n说明:\n 首帧延迟 = 用户说完到听到第一个音频输出\n LLM-TTFB = LLM首个token返回时间\n ASR召回 = 领域术语在ASR结果中的命中率")


if __name__ == "__main__":
    manifest = "test_manifest.json"
    if len(sys.argv) > 1:
        manifest = sys.argv[1]

    print("初始化在线API引擎…")
    asr = StreamingASR(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
    llm = StreamingLLM(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
    tts = StreamingTTS(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, TTS_VOICE)
    init_pipelines(asr, llm, tts)

    asyncio.run(run_experiment(manifest))
