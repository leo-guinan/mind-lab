"""Quick concurrency test: how many parallel OpenRouter calls can we sustain?"""
import subprocess, os, requests, time, concurrent.futures

# Get key
result = subprocess.run(['grep', 'OPENROUTER_API_KEY', os.path.expanduser('~/.hermes/.env')], capture_output=True, text=True)
key = result.stdout.strip().split('=', 1)[1].strip().strip("'").strip('"')

def call_once(i):
    start = time.time()
    resp = requests.post('https://openrouter.ai/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        json={'model': 'google/gemini-2.5-flash-lite', 'messages': [{'role': 'user', 'content': 'Say "ok".'}], 'max_tokens': 5},
        timeout=30)
    elapsed = time.time() - start
    status = resp.status_code
    # Check rate limit headers
    limit = resp.headers.get('x-ratelimit-requests-limit', '?')
    remaining = resp.headers.get('x-ratelimit-requests-remaining', '?')
    reset = resp.headers.get('x-ratelimit-requests-reset', '?')
    return i, elapsed, status, limit, remaining, reset

# Test with 5 concurrent, then 10, then 20
for concurrency in [5, 10, 20]:
    print(f'\n=== {concurrency} concurrent calls ===')
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(call_once, i) for i in range(concurrency)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    total_t = time.time() - t0
    
    times = [r[1] for r in results]
    statuses = [r[2] for r in results]
    limits = set((r[3], r[4], r[5]) for r in results if r[3] != '?')
    
    ok = sum(1 for s in statuses if s == 200)
    err = len(statuses) - ok
    avg = sum(times) / len(times) if times else 0
    print(f'  OK: {ok}/{concurrency}  Errors: {err}  Wall time: {total_t:.1f}s  Avg latency: {avg:.1f}s')
    if limits:
        for l in limits:
            print(f'  Rate limit: req_limit={l[0]} remaining={l[1]} reset={l[2]}')
    if err:
        print(f'  Status codes: {set(statuses)}')
