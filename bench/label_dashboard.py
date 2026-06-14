#!/usr/bin/env python3
"""
Mind Lab Year-by-Year Labeling Dashboard

Shows live progress bars for each year (2020-2026), broken into monthly cells
that fill from grey (queued) → yellow (processing) → green (complete).

Workers pull from all year queues simultaneously, calling OpenRouter gemini-2.5-flash-lite.
"""

import json, os, subprocess, time, pathlib, threading, queue, sys
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Group
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# === CONFIG ===

DATA_PATH = pathlib.Path('/Users/leoguinan/Projects/mind-lab/data/tweets_raw.jsonl')
OUTPUT_PATH = pathlib.Path('/Users/leoguinan/Projects/mind-lab/data/labeled.jsonl')
CHECKPOINT_PATH = pathlib.Path('/Users/leoguinan/Projects/mind-lab/data/labeled_checkpoint.json')
MODEL = 'google/gemini-2.5-flash-lite'
WORKERS = 12

# Load OpenRouter key
result = subprocess.run(['grep', 'OPENROUTER_API_KEY', os.path.expanduser('~/.hermes/.env')], capture_output=True, text=True)
OR_KEY = result.stdout.strip().split('=', 1)[1].strip().strip("'").strip('"')

# Labeling prompt
LABEL_PROMPT = """Classify this tweet into functional components. Return ONLY valid JSON with weights 0-100.

Components:
- original_theory: introduces a new model, framework, or causal claim
- observation: describes a pattern, phenomenon, without theorizing
- synthesis: connects multiple ideas, threads, or domains
- meta_analysis: reflects on thinking itself, second-order analysis
- research: asks questions, gathers data, investigates systematically
- teaching: explains a concept for others to understand
- personal_story: shares an experience, anecdote, or personal journey
- prediction: forecasts a future outcome or trend
- coordination: calls to action, organizing people, collaborative intent
- news: shares or comments on current events
- coaching: offers guidance, feedback, or actionable advice

For each component that applies, give weight (0-100) and a brief reason.

Tweet: {text}

Respond ONLY with: {{"components": {{"component_name": {{"weight": 80, "reason": "..."}}}}}}"""


# === DATA PREP ===

def load_tweets():
    """Load tweets, filter, bucketed by year and month."""
    buckets = defaultdict(lambda: defaultdict(list))
    all_tweets = []
    
    with DATA_PATH.open() as f:
        for line in f:
            t = json.loads(line)
            text = t.get('full_text', '')
            if text.startswith('RT '):
                continue
            if len(text) < 50:
                continue
            created = t.get('created_at', '')
            if not created:
                continue
            year = created[:4]
            month = created[5:7]
            tweet_obj = {'tweet_id': t['tweet_id'], 'text': text, 'year': year, 'month': month}
            buckets[year][month].append(tweet_obj)
            all_tweets.append(tweet_obj)
    
    return buckets, all_tweets


def load_checkpoint():
    """Load already-labeled tweet IDs so we can resume."""
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open() as f:
            return set(json.load(f))
    return set()


def save_checkpoint(labeled_ids):
    with CHECKPOINT_PATH.open('w') as f:
        json.dump(list(labeled_ids), f)


# === API CALL ===

def label_tweet(text):
    """Call OpenRouter for one tweet. Returns (success, parsed_json_or_error, elapsed_s)."""
    start = time.time()
    try:
        resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization': f'Bearer {OR_KEY}', 'Content-Type': 'application/json'},
            json={
                'model': MODEL,
                'messages': [{'role': 'user', 'content': LABEL_PROMPT.replace('{text}', text)}],
                'temperature': 0.1,
                'max_tokens': 500,
            },
            timeout=60,
        )
        elapsed = time.time() - start
        data = resp.json()
        if 'error' in data:
            return False, data['error'].get('message', str(data['error'])), elapsed
        
        raw = data['choices'][0]['message']['content']
        # Parse JSON
        try:
            if '```' in raw:
                raw = raw.split('```')[1].split('```')[0]
                if raw.startswith('json'):
                    raw = raw[4:]
            if '{' in raw:
                raw = raw[raw.index('{'):raw.rindex('}')+1]
            parsed = json.loads(raw)
            return True, parsed, elapsed
        except Exception:
            return False, f'parse error: {raw[:100]}', elapsed
    except Exception as e:
        elapsed = time.time() - start
        return False, str(e)[:100], elapsed


# === SHARED STATE ===

class LabelingState:
    """Thread-safe shared state for the dashboard."""
    
    def __init__(self, buckets, completed_ids):
        self.lock = threading.RLock()
        self.completed_ids = set(completed_ids)
        
        # Year -> Month -> [tweets]
        self.tweets = {}
        # Year -> Month -> done_count
        self.done = defaultdict(lambda: defaultdict(int))
        # Year -> Month -> total_count
        self.totals = defaultdict(lambda: defaultdict(int))
        # Year -> Month -> active (being processed)
        self.active = defaultdict(lambda: defaultdict(bool))
        
        for year, months in buckets.items():
            for month, tweet_list in months.items():
                total = len(tweet_list)
                self.totals[year][month] = total
                remaining = []
                for t in tweet_list:
                    if t['tweet_id'] in completed_ids:
                        self.done[year][month] += 1
                    else:
                        remaining.append(t)
                self.tweets.setdefault(year, {})[month] = remaining
        
        # Global queue
        self.queue = queue.Queue()
        for year, months in self.tweets.items():
            for month, tweet_list in months.items():
                for t in tweet_list:
                    self.queue.put(t)
        
        self.total_tweets = self.queue.qsize() + sum(
            self.done[y][m] for y in self.done for m in self.done[y]
        )
        self.done_total = sum(self.done[y][m] for y in self.done for m in self.done[y])
        self.start_time = time.time()
        self.recent_times = []  # Last 20 completion times for ETA
        self.errors = []
        self.running = True
        self.results = []  # Store results for output
        self.num_workers = 0  # Set by main()
    
    def record_done(self, year, month):
        with self.lock:
            self.done[year][month] += 1
            self.done_total += 1
            self.recent_times.append(time.time())
            if len(self.recent_times) > 50:
                self.recent_times = self.recent_times[-50:]
            self.active[year][month] = True
    
    def record_error(self, err):
        with self.lock:
            self.errors.append(err[:80])
    
    def get_eta(self):
        with self.lock:
            if len(self.recent_times) < 2:
                return None
            remaining = self.queue.qsize()
            if remaining == 0:
                return 0
            # Rate from last 20 completions
            recent = self.recent_times[-20:]
            if len(recent) < 2:
                return None
            rate = (len(recent) - 1) / max(recent[-1] - recent[0], 0.001)
            return remaining / rate if rate > 0 else None
    
    def set_active(self, year, month):
        with self.lock:
            self.active[year][month] = True
    
    def clear_active(self, year, month):
        with self.lock:
            self.active[year][month] = False


# === DASHBOARD ===

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def build_dashboard(state):
    """Build a rich renderable for the dashboard."""
    table = Table(
        title=f"[bold white]Mind Lab — Labeling Dashboard[/]  [dim]{MODEL}[/dim]",
        title_style="bold cyan",
        border_style="dim blue",
        padding=(0, 1),
    )
    
    table.add_column("Year", style="bold white", width=6, justify="right")
    table.add_column("Progress", width=8, justify="right")
    for m in MONTH_NAMES:
        table.add_column(m, width=3, justify="center")
    table.add_column("Done", width=6, justify="right")
    
    years = ['2020', '2021', '2022', '2023', '2024', '2025', '2026']
    
    for year in years:
        with state.lock:
            year_done = sum(state.done[year][f'{i+1:02d}'] for i in range(12))
            year_total = sum(state.totals[year][f'{i+1:02d}'] for i in range(12))
            pct = year_done / year_total * 100 if year_total else 100
        
        if year_total == 0:
            continue
        
        # Progress bar
        bar_width = 8
        filled = int(bar_width * year_done / year_total) if year_total else bar_width
        bar = f"[green]{'█' * filled}[/][dim]{'░' * (bar_width - filled)}[/]"
        pct_str = f"{year_done}/{year_total}"
        
        row = [year, f"{bar} {pct_str}"]
        
        for i, m in enumerate(MONTH_NAMES):
            month_key = f'{i+1:02d}'
            with state.lock:
                done = state.done[year][month_key]
                total = state.totals[year].get(month_key, 0)
                is_active = state.active[year][month_key]
            
            if total == 0:
                cell = "[dim]·[/]"
            elif done >= total:
                cell = "[bold green]█[/]"
            elif is_active:
                cell = "[bold yellow]▓[/]"
            elif done > 0:
                cell = "[blue]░[/]"
            else:
                cell = "[dim]░[/]"
            row.append(cell)
        
        row.append(str(year_done))
        table.add_row(*row)
    
    # Stats footer
    with state.lock:
        done = state.done_total
        remaining = state.queue.qsize()
        total = done + remaining
        elapsed = time.time() - state.start_time
        eta = state.get_eta()
        errors = len(state.errors)
    
    elapsed_str = f"{int(elapsed//60)}m{int(elapsed%60)}s"
    eta_str = f"{int(eta//60)}m{int(eta%60)}s" if eta else "calculating..."
    rate = done / elapsed if elapsed > 0 else 0
    
    footer = Text()
    footer.append(f"\n[bold]{done:,}/{total:,}[/] tweets labeled ")
    footer.append(f"({done/total*100:.1f}%)  ", style="green")
    footer.append(f"│  [bold]{rate:.1f}[/] tweets/sec  ", style="cyan")
    footer.append(f"│  elapsed: {elapsed_str}  ")
    footer.append(f"│  ETA: {eta_str}  ")
    footer.append(f"│  workers: {state.num_workers}  ")
    if errors:
        footer.append(f"│  [red]errors: {errors}[/]")
    
    return Group(table, footer)


# === WORKER ===

def worker(state, worker_id):
    """Worker thread: pull from global queue, label, update state."""
    while state.running:
        try:
            tweet = state.queue.get(timeout=1)
        except queue.Empty:
            continue
        except Exception as e:
            print(f'[W{worker_id}] queue.get error: {e}', flush=True)
            continue
        
        year = tweet['year']
        month = tweet['month']
        tid = tweet['tweet_id']
        
        state.set_active(year, month)
        try:
            success, result, elapsed = label_tweet(tweet['text'])
        except Exception as e:
            print(f'[W{worker_id}] label exception: {e}', flush=True)
            state.queue.put(tweet)  # requeue
            state.clear_active(year, month)
            state.queue.task_done()
            continue
        
        if success:
            state.results.append({
                'tweet_id': tid,
                'year': year,
                'month': month,
                'label': result,
                'elapsed_s': round(elapsed, 2),
            })
            state.record_done(year, month)
            
            # Save incrementally (every 50)
            if len(state.results) % 50 == 0:
                save_results(state)
                save_checkpoint(set(r['tweet_id'] for r in state.results))
        else:
            state.record_error(result)
            # Re-queue once
            if 'rate' not in str(result).lower():
                state.queue.put(tweet)
        
        state.clear_active(year, month)
        state.queue.task_done()


def save_results(state):
    """Save results to JSONL."""
    existing = []
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open() as f:
            for line in f:
                try:
                    existing.append(json.loads(line))
                except:
                    pass
    
    all_results = {r['tweet_id']: r for r in existing}
    for r in state.results:
        all_results[r['tweet_id']] = r
    
    with OUTPUT_PATH.open('w') as f:
        for r in all_results.values():
            f.write(json.dumps(r) + '\n')


# === MAIN ===

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=0, help='Max tweets to process (0 = all)')
    parser.add_argument('--workers', type=int, default=12, help='Number of workers')
    args = parser.parse_args()
    
    num_workers = args.workers
    max_tweets = args.max
    
    print("Loading tweets...")
    buckets, all_tweets = load_tweets()
    completed_ids = load_checkpoint()
    
    print(f"Resuming from checkpoint: {len(completed_ids)} already labeled")
    
    state = LabelingState(buckets, completed_ids)
    state.num_workers = num_workers
    remaining = state.queue.qsize()
    
    # Apply limit
    if max_tweets and max_tweets < remaining:
        # Trim queue
        trimmed = []
        for _ in range(max_tweets):
            try:
                t = state.queue.get_nowait()
                trimmed.append(t)
            except queue.Empty:
                break
        state.queue = queue.Queue()
        for t in trimmed:
            state.queue.put(t)
        state.total_tweets = len(completed_ids) + state.queue.qsize()
        remaining = state.queue.qsize()
    
    print(f"Tweets to label: {state.total_tweets:,} ({remaining:,} remaining)")
    print(f"Workers: {num_workers}")
    print(f"Model: {MODEL}")
    print()
    
    if remaining == 0:
        print("All tweets already labeled!")
        return
    
    # Start workers
    is_tty = sys.stdout.isatty()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, state, i) for i in range(num_workers)]
        
        if is_tty:
            # Live TUI mode
            with Live(build_dashboard(state), refresh_per_second=4, screen=True) as live:
                while state.queue.qsize() > 0:
                    live.update(build_dashboard(state))
                    time.sleep(0.25)
                state.running = False
                live.update(build_dashboard(state))
        else:
            # Print mode (background/non-TTY)
            while state.queue.qsize() > 0:
                time.sleep(2)
                with state.lock:
                    done = state.done_total
                    remaining = state.queue.qsize()
                    pct = done / (done + remaining) * 100 if (done + remaining) else 100
                    elapsed = time.time() - state.start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = state.get_eta()
                eta_s = f'{int(eta//60)}m{int(eta%60)}s' if eta else '...'
                print(f'\r[{pct:.0f}%] {done}/{done+remaining} tweets | {rate:.1f}/s | ETA {eta_s}', end='', flush=True)
            state.running = False
            print()
    
    # Save final results
    save_results(state)
    save_checkpoint(set(r['tweet_id'] for r in state.results))
    
    with state.lock:
        errors = len(state.errors)
    
    print(f"\n✓ Done! {state.done_total:,} tweets labeled.")
    print(f"  Results: {OUTPUT_PATH}")
    print(f"  Errors: {errors}")


if __name__ == '__main__':
    main()
