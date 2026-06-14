#!/usr/bin/env python3
"""Benchmark Ollama vs OpenRouter for tweet functional labeling."""

import json, time, os, pathlib, subprocess, sys
from datetime import datetime
import requests

BENCH = pathlib.Path('/Users/leoguinan/Projects/mind-lab/bench')
TEST_SET = BENCH / 'benchmark_set.json'
RESULTS = BENCH / 'bench_results.json'
COMPARISON = BENCH / 'bench_comparison.md'

# 11-component taxonomy
LABEL_PROMPT = """Classify this tweet into functional components. Return JSON with weights 0-100 for each component that applies (sum need not be 100).

Components defined:
- original_theory: introduces a new model, framework, or causal claim about how something works
- observation: describes a pattern, phenomenon, or data point without theorizing
- synthesis: connects multiple ideas, threads, or domains into a coherent narrative
- meta_analysis: reflects on thinking itself, second-order analysis of methods/approaches
- research: asks questions, gathers data, or investigates systematically
- teaching: explains a concept for others to understand, educational intent
- personal_story: shares an experience, anecdote, or personal journey
- prediction: forecasts a future outcome or trend with reasoning
- coordination: calls to action, organizing people, or collaborative intent
- news: shares or comments on current events, announcements, or developments
- coaching: offers guidance, feedback, or actionable advice directed at others

For each component that applies, provide weight (0-100) and a brief reason.

Tweet: {text}

Respond with ONLY valid JSON like:
{"components": {"original_theory": {"weight": 80, "reason": "introduces new causal model"}, "teaching": {"weight": 40, "reason": "explains concept"}}, "dominant": "original_theory"}
"""

# OpenRouter key
with open(os.path.expanduser('~/.hermes/.env')) as f:
    for line in f:
        if line.startswith('OPENROUTER_API_KEY='):
            OPENROUTER_KEY = line.split('=', 1)[1].strip().strip("'").strip('"')
            break

# Models to benchmark
MODELS = {
    # Ollama (free, local)
    'ollama-llama3.2': {'type': 'ollama', 'model': 'llama3.2', 'cost_in': 0, 'cost_out': 0},
    'ollama-llama3.1-8b': {'type': 'ollama', 'model': 'llama3.1:8b', 'cost_in': 0, 'cost_out': 0},
    'ollama-qwen3-4b': {'type': 'ollama', 'model': 'qwen3:4b', 'cost_in': 0, 'cost_out': 0},
    # OpenRouter (priced)
    'openrouter-deepseek-v3': {'type': 'openrouter', 'model': 'deepseek/deepseek-chat', 'cost_in': 0.27, 'cost_out': 1.10},
    'openrouter-qwen2.5-7b': {'type': 'openrouter', 'model': 'qwen/qwen-2.5-7b-instruct', 'cost_in': 0.07, 'cost_out': 0.16},
    'openrouter-gemini-flash': {'type': 'openrouter', 'model': 'google/gemini-2.0-flash-001', 'cost_in': 0.10, 'cost_out': 0.40},
}

def ollama_label(model_name, text):
    """Call Ollama for labeling."""
    start = time.time()
    try:
        resp = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,
                'prompt': LABEL_PROMPT.replace('{text}', text),
                'stream': False,
                'options': {'temperature': 0.1, 'num_predict': 500},
            },
            timeout=120,
        )
        elapsed = time.time() - start
        data = resp.json()
        # Extract output
        output = data.get('response', '')
        tokens_in = data.get('prompt_eval_count', None)
        tokens_out = data.get('eval_count', None)
        return {
            'output': output,
            'elapsed_s': round(elapsed, 2),
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'tokens_per_sec': round(tokens_out / elapsed, 1) if tokens_out and elapsed > 0 else None,
            'error': None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'output': None,
            'elapsed_s': round(elapsed, 2),
            'tokens_in': None,
            'tokens_out': None,
            'tokens_per_sec': None,
            'error': str(e),
        }

def openrouter_label(model_id, text):
    """Call OpenRouter for labeling."""
    start = time.time()
    try:
        resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENROUTER_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': model_id,
                'messages': [{'role': 'user', 'content': LABEL_PROMPT.replace('{text}', text)}],
                'temperature': 0.1,
                'max_tokens': 500,
            },
            timeout=120,
        )
        elapsed = time.time() - start
        data = resp.json()
        if 'error' in data:
            return {
                'output': None,
                'elapsed_s': round(elapsed, 2),
                'tokens_in': None,
                'tokens_out': None,
                'tokens_per_sec': None,
                'error': data['error'].get('message', str(data['error'])),
            }
        choice = data['choices'][0]
        output = choice['message']['content']
        usage = data.get('usage', {})
        tokens_in = usage.get('prompt_tokens')
        tokens_out = usage.get('completion_tokens')
        return {
            'output': output,
            'elapsed_s': round(elapsed, 2),
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'tokens_per_sec': round(tokens_out / elapsed, 1) if tokens_out and elapsed > 0 else None,
            'error': None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'output': None,
            'elapsed_s': round(elapsed, 2),
            'tokens_in': None,
            'tokens_out': None,
            'tokens_per_sec': None,
            'error': str(e),
        }

def run_benchmark():
    """Run full benchmark."""
    with open(TEST_SET) as f:
        tweets = json.load(f)
    
    results = {}
    total_tweets = len(tweets)
    
    for name, config in MODELS.items():
        print(f'\n=== {name} ({config["model"]}) ===')
        model_results = []
        cum_time = 0
        cum_cost = 0
        cum_tokens_in = 0
        cum_tokens_out = 0
        errors = 0
        
        for i, tweet in enumerate(tweets):
            text = tweet['text']
            tid = tweet['tweet_id'][-8:]
            print(f'  [{i+1}/{total_tweets}] {tid}...', end=' ', flush=True)
            
            if config['type'] == 'ollama':
                r = ollama_label(config['model'], text)
            else:
                r = openrouter_label(config['model'], text)
            
            parsed = None
            if r['error']:
                print(f'ERR: {r["error"][:60]}')
                errors += 1
            else:
                # Try to parse JSON from output
                try:
                    raw = r['output']
                    # Extract JSON if wrapped in markdown
                    if '```json' in raw:
                        raw = raw.split('```json')[1].split('```')[0]
                    elif '```' in raw:
                        raw = raw.split('```')[1].split('```')[0]
                    elif '{' in raw:
                        raw = raw[raw.index('{'):raw.rindex('}')+1]
                    parsed = json.loads(raw)
                except Exception:
                    pass
                
                tokens_s = f'{r["tokens_in"]}+{r["tokens_out"]}tok' if r['tokens_in'] else '?tok'
                print(f'{r["elapsed_s"]}s {tokens_s} {"✓" if parsed else "✗"}')
                
                cum_time += r['elapsed_s']
                cum_tokens_in += r['tokens_in'] or 0
                cum_tokens_out += r['tokens_out'] or 0
                # Cost: per million tokens
                cost = ((r['tokens_in'] or 0) * config['cost_in'] + (r['tokens_out'] or 0) * config['cost_out']) / 1_000_000
                cum_cost += cost
            
            model_results.append({
                'tweet_id': tweet['tweet_id'],
                'elapsed_s': r['elapsed_s'],
                'tokens_in': r['tokens_in'],
                'tokens_out': r['tokens_out'],
                'tokens_per_sec': r['tokens_per_sec'],
                'output_parsed': parsed is not None,
                'error': r['error'],
                'raw_output': r['output'][:500] if r['output'] else None,
            })
            
            # Small delay to avoid rate limits
            time.sleep(0.3)
        
        results[name] = {
            'model': config['model'],
            'type': config['type'],
            'cost_per_1M_in': config['cost_in'],
            'cost_per_1M_out': config['cost_out'],
            'total_tweets': total_tweets,
            'errors': errors,
            'total_time_s': round(cum_time, 1),
            'avg_time_s': round(cum_time / total_tweets, 2),
            'total_tokens_in': cum_tokens_in,
            'total_tokens_out': cum_tokens_out,
            'avg_tokens_in': round(cum_tokens_in / total_tweets) if total_tweets else 0,
            'avg_tokens_out': round(cum_tokens_out / total_tweets) if total_tweets else 0,
            'total_cost_usd': round(cum_cost, 6),
            'cost_per_1k_tweets': round(cum_cost / total_tweets * 1000, 4),
            'results': model_results,
        }
        
        print(f'  TOTAL: {round(cum_time,1)}s | ${round(cum_cost,6)} | {total_tweets-errors}/{total_tweets} ok')
    
    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    results = run_benchmark()
    print(f'\nResults saved to {RESULTS}')
