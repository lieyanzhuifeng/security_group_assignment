"""
W1 vs W2 首段音频延迟对比实验

度量: 从「用户端停止说话」(音频就绪) 到「首段音频开始播放」的时间

关键指标: TimingMetrics.first_audio
  - W1: ASR + 完整LLM + 完整TTS (顺序执行，first_audio ≈ total)
  - W2: ASR + 首个句子LLM + 首个句子TTS (流式切分，first_audio 远小于 total)
"""

import asyncio
import json
import time
import sys
from pathlib import Path

from config import *
from streaming.asr_azure import StreamingASR
from streaming.llm_api import StreamingLLM
from streaming.tts_azure import StreamingTTS
from pipelines.factory import init_pipelines, get_pipeline

TESTS_DIR = Path(__file__).parent / "tests"
MANIFEST_PATH = Path(__file__).parent / "test_manifest.json"

N_RUNS = 3
SEPARATOR = "=" * 72


def load_test_cases():
    """加载存在且 safe 的测试用例"""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cases = []
    for t in manifest:
        wav_path = TESTS_DIR / Path(t["audio"]).name
        if wav_path.exists() and t.get("safe", True):
            cases.append((t["id"], wav_path))
        elif not wav_path.exists():
            print(f"[SKIP] {t['id']}: 音频文件不存在 - {wav_path}")
    return cases


async def run_one(pipeline, audio_bytes: bytes, label: str, run_id: int):
    """运行一次 pipeline，返回 (first_audio, asr, llm, llm_ttfb, total, error)"""
    t0 = time.perf_counter()
    result = await pipeline.run(audio_bytes)
    wall = time.perf_counter() - t0

    t = result.timings
    first_audio = t.first_audio if t.first_audio is not None else t.total

    if result.error:
        print(f"  [{label}] run {run_id}: ERROR - {result.error}")
        return None

    print(f"  [{label}] run {run_id}: "
          f"first_audio={first_audio:.2f}s  "
          f"asr={t.asr:.2f}s  llm_ttfb={t.llm_ttfb:.2f}s  "
          f"total={t.total:.2f}s  wall={wall:.2f}s")
    return first_audio


async def main():
    cases = load_test_cases()
    if not cases:
        print("没有可用的测试用例")
        return

    asr = StreamingASR(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
    llm = StreamingLLM(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
    tts_ = StreamingTTS(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, TTS_VOICE)
    init_pipelines(asr, llm, tts_)

    w1 = get_pipeline("w1")
    w2 = get_pipeline("w2")

    print(f"\n{SEPARATOR}")
    print(f"  W1 vs W2  首段音频延迟对比实验")
    print(f"  度量: 音频就绪 → 首段TTS音频合成完成")
    print(f"  每个用例运行 {N_RUNS} 次取平均")
    print(f"{SEPARATOR}\n")

    all_results = []

    for test_id, wav_path in cases:
        audio_bytes = wav_path.read_bytes()
        file_size_kb = len(audio_bytes) / 1024

        print(f"{'─' * 60}")
        print(f"  {test_id}: {wav_path.name}  ({file_size_kb:.0f} KB)")
        print(f"{'─' * 60}")

        w1_times = []
        w2_times = []

        for i in range(1, N_RUNS + 1):
            r1 = await run_one(w1, audio_bytes, "W1", i)
            if r1 is not None:
                w1_times.append(r1)

            r2 = await run_one(w2, audio_bytes, "W2", i)
            if r2 is not None:
                w2_times.append(r2)

        if w1_times and w2_times:
            w1_avg = sum(w1_times) / len(w1_times)
            w2_avg = sum(w2_times) / len(w2_times)
            speedup = w1_avg / w2_avg if w2_avg > 0 else 0
            delta = w1_avg - w2_avg

            all_results.append({
                "id": test_id,
                "w1_avg": w1_avg,
                "w2_avg": w2_avg,
                "speedup": speedup,
                "delta": delta,
            })

            print(f"  {'─' * 40}")
            print(f"  > W1 avg = {w1_avg:.2f}s  |  W2 avg = {w2_avg:.2f}s  "
                  f"|  加速 {speedup:.1f}x  (节省 {delta:.1f}s)")
        else:
            print(f"  > 数据不足，跳过")

        print()

    if not all_results:
        print("没有足够的有效数据")
        return

    print(f"{SEPARATOR}")
    print(f"  {'汇总':^50}")
    print(f"{SEPARATOR}")
    print(f"  {'用例':<8} {'W1 (s)':>8} {'W2 (s)':>8} {'加速 (x)':>9} {'节省 (s)':>9}")
    print(f"  {'─' * 50}")
    for r in all_results:
        print(f"  {r['id']:<8} {r['w1_avg']:>8.2f} {r['w2_avg']:>8.2f} "
              f"{r['speedup']:>8.1f}x {r['delta']:>8.1f}s")

    overall_w1 = sum(r["w1_avg"] for r in all_results) / len(all_results)
    overall_w2 = sum(r["w2_avg"] for r in all_results) / len(all_results)
    overall_speedup = overall_w1 / overall_w2 if overall_w2 > 0 else 0

    print(f"  {'─' * 50}")
    print(f"  {'平均':<8} {overall_w1:>8.2f} {overall_w2:>8.2f} "
          f"{overall_speedup:>8.1f}x {overall_w1 - overall_w2:>8.1f}s")
    print(f"{SEPARATOR}\n")


if __name__ == "__main__":
    asyncio.run(main())
