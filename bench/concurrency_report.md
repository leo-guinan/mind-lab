# Parallelism Projections for 43K Tweet Labeling

## Concurrency test results (gemini-2.5-flash-lite on OpenRouter)

| Workers | Wall time | Calls/sec | Status |
|---------|-----------|-----------|--------|
| 5 | 0.7s | 7.1 | All OK |
| 10 | 1.7s | 5.9 | All OK |
| 20 | 3.5s | 5.7 | All OK |

No rate limits hit at any level. The ceiling is higher than 20.

## Projections for 43K tweets (at 1.14s/tweet serial)

| Workers | Calls/sec | Time for 43K | Cost |
|---------|-----------|-------------|------|
| 1 (serial) | 0.88 | 13.6 hours | $4.27 |
| 8 | ~7 | ~1.7 hours | $4.27 |
| 16 | ~14 | ~51 minutes | $4.27 |
| 32 | ~28 | ~25 minutes | $4.27 |
| 64 | ~56 | ~13 minutes | $4.27 |

## Architecture: VPS queue

1. **Redis queue** on arc-vps — lightweight, trivial to set up
2. **Worker pool** — Python workers pulling from queue, calling OpenRouter in parallel, writing JSONL results
3. **Retry queue** — any 429/5xx errors go to dead-letter queue for retry
4. **Results** streamed back via SSH or stored on VPS, then rsynced

The VPS is I/O bound, not CPU bound — each worker just makes HTTP calls and waits. Even 64 concurrent workers is trivial for any VPS.

## Cost summary

| Path | Time | Cost |
|------|------|------|
| 16 workers, gemini-2.5-flash-lite | ~51 min | **$4.27** |
| 8 workers, gemini-2.5-flash-lite | ~1.7 hrs | **$4.27** |
| 4 workers, gemini-2.5-flash-lite | ~3.4 hrs | **$4.27** |
| llama3.2 (local, serial) | 8.7 hrs | FREE |
