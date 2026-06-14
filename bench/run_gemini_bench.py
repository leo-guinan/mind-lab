"""Run single-model benchmark for Gemini Flash Lite and append to results."""
import json, os, requests, time, pathlib, subprocess

BENCH = pathlib.Path('/Users/leoguinan/Projects/mind-lab/bench')

# Load key via env
result = subprocess.run(['grep', 'OPENROUTER_API_KEY', os.path.expanduser('~/.hermes/.env')], capture_output=True, text=True)
OR_KEY = result.stdout.strip().split('=', 1)[1].strip().strip("'").strip('"')

# Load prompt template
with open(BENCH / 'benchmark.py') as f:
    bench_code = f.read()
# Extract the LABEL_PROMPT string
import re
m = re.search(r'LABEL_PROMPT = """(.+?)"""', bench_code, re.DOTALL)
if m is None:
    # Try importing directly from the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("benchmark", str(BENCH / 'benchmark.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    LABEL_PROMPT = mod.LABEL_PROMPT
else:
    LABEL_PROMPT = m.group(1)

# Load tweets
with open(BENCH / 'benchmark_set.json') as f:
    tweets = json.load(f)

# Model config
model_id = 'google/gemini-2.5-flash-lite'
cost_in = 0.10   # USD per 1M tokens
cost_out = 0.40

print(f'=== openrouter-gemini-2.5-flash-lite ({model_id}) ===')
results = []
cum_time = 0
cum_cost = 0
cum_tokens_in = 0
cum_tokens_out = 0
errors = 0
total = len(tweets)

for i, tweet in enumerate(tweets):
    text = tweet['text']
    tid = tweet['tweet_id'][-8:]
    print(f'  [{i+1}/{total}] {tid}...', end=' ', flush=True)
    
    start = time.time()
    try:
        resp = requests.post('https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization': f'Bearer {OR_KEY}', 'Content-Type': 'application/json'},
            json={
                'model': model_id,
                'messages': [{'role': 'user', 'content': LABEL_PROMPT.replace('{text}', text)}],
                'temperature': 0.1,
                'max_tokens': 500,
            },
            timeout=120)
        elapsed = time.time() - start
        data = resp.json()
        
        if 'error' in data:
            err_msg = data['error'].get('message', str(data['error']))
            print(f'ERR: {err_msg[:60]}')
            errors += 1
            results.append({'tweet_id': tweet['tweet_id'], 'elapsed_s': round(elapsed, 2), 'error': err_msg})
            time.sleep(0.3)
            continue
        
        choice = data['choices'][0]
        output = choice['message']['content']
        usage = data.get('usage', {})
        tokens_in = usage.get('prompt_tokens', 0)
        tokens_out = usage.get('completion_tokens', 0)
        tps = round(tokens_out / elapsed, 1) if tokens_out and elapsed > 0 else None
        
        # Parse JSON from output
        parsed = None
        raw = output
        try:
            if '```json' in raw:
                extracted = raw.split('```json')[1].split('```')[0]
            elif '```' in raw:
                extracted = raw.split('```')[1].split('```')[0]
            elif '{' in raw:
                extracted = raw[raw.index('{'):raw.rindex('}')+1]
            else:
                extracted = raw
            parsed = json.loads(extracted)
        except Exception:
            pass
        
        cost = (tokens_in * cost_in + tokens_out * cost_out) / 1_000_000
        cum_cost += cost
        cum_time += elapsed
        cum_tokens_in += tokens_in
        cum_tokens_out += tokens_out
        
        print(f'{elapsed:.1f}s {tokens_in}+{tokens_out}tok {"✓" if parsed else "✗"}')
        results.append({
            'tweet_id': tweet['tweet_id'], 'elapsed_s': round(elapsed, 2),
            'tokens_in': tokens_in, 'tokens_out': tokens_out,
            'tokens_per_sec': tps, 'output_parsed': parsed is not None,
            'error': None, 'raw_output': output[:500] if output else None,
        })
    except Exception as e:
        elapsed = time.time() - start
        print(f'ERR: {str(e)[:60]}')
        errors += 1
        results.append({'tweet_id': tweet['tweet_id'], 'elapsed_s': round(elapsed, 2), 'error': str(e)})
    
    time.sleep(0.3)

summary = {
    'model': model_id, 'type': 'openrouter',
    'cost_per_1M_in': cost_in, 'cost_per_1M_out': cost_out,
    'total_tweets': total, 'errors': errors,
    'total_time_s': round(cum_time, 1), 'avg_time_s': round(cum_time/total, 2),
    'total_tokens_in': cum_tokens_in, 'total_tokens_out': cum_tokens_out,
    'avg_tokens_in': round(cum_tokens_in/total), 'avg_tokens_out': round(cum_tokens_out/total),
    'total_cost_usd': round(cum_cost, 6),
    'cost_per_1k_tweets': round(cum_cost/total*1000, 4),
    'results': results,
}
print(f'  TOTAL: {round(cum_time,1)}s | ${round(cum_cost,6)} | {total-errors}/{total} ok')

# Load existing results, add this one, save
results_path = BENCH / 'bench_results.json'
with open(results_path) as f:
    all_results = json.load(f)
all_results['openrouter-gemini-2.5-flash-lite'] = summary
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\nAppended to {results_path}')
