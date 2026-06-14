# Tweet Labeling Benchmark: Ollama vs OpenRouter

21 diverse tweets from 2021–2026, labeled across 5 working models.

## Raw Results

| Model | Type | Cost | Time (21) | Time/tweet | Parse Rate | Speed |
|-------|------|------|-----------|------------|------------|-------|
| llama3.2 (2GB) | Ollama | FREE | 15.4s | 0.73s | 95% | 77 tok/s |
| llama3.1:8b (5GB) | Ollama | FREE | 40.6s | 1.93s | 100% | 48 tok/s |
| qwen3:4b (2.5GB) | Ollama | FREE | 124.9s | 5.95s | 0% | 87 tok/s |
| deepseek-v3 | OpenRouter | $0.196/K | 94.4s | 4.49s | 100% | 22 tok/s |
| qwen2.5-7b | OpenRouter | $0.034/K | 135.4s | 6.45s | 100% | 10 tok/s |
| gemini-2.0-flash | OpenRouter | — | FAIL | — | — | — |

## Projected for 43K tweets

| Model | Total Time | Total Cost | Labels Produced | Verdict |
|-------|-----------|------------|-----------------|---------|
| llama3.2 (2GB) | 8.7 hrs | FREE | ~41K (95%) | Speed king, 5% json errors |
| llama3.1:8b (5GB) | 23.1 hrs | FREE | 43K (100%) | Best balance, zero failures |
| qwen3:4b (2.5GB) | — | — | 0 (0%) | Useless for this task |
| deepseek-v3 | 53.7 hrs | $8.40 | 43K (100%) | Best quality, slow serial |
| qwen2.5-7b | 77.1 hrs | $1.46 | 43K (100%) | Cheapest paid, slowest |

## Recommendation

**Use Ollama llama3.1:8b locally.** Free, 100% parse rate, 23 hours serial. 

If you want it done faster, run 4 concurrent Ollama instances → ~6 hours, still free.
If you want it done NOW, run 8 concurrent qwen2.5-7b calls via OpenRouter → ~10 hours, ~$1.46.

## Parsed output quality

All 100%-parse models produced structurally identical JSON with per-component weights and reasons. llama3.1:8b's output was as coherent as deepseek-v3's on these 21 samples. The free model keeps up with the $0.196/K one on this task.

## Notes

- gemini-2.0-flash model ID is wrong — needs updating
- qwen3:4b outputs unparseable text. Dead model for structured tasks.
- Times are serial (single-threaded). Real throughput can be 4-8x with parallel calls.
