"""Analyze and compare benchmark results."""

import json
from pathlib import Path
import pandas as pd

def load_results(results_dir: str = "results/raw_results") -> pd.DataFrame:
    """Load all benchmark results into a DataFrame."""
    results_dir = Path(results_dir)
    
    all_results = []
    for json_file in sorted(results_dir.glob("benchmark_*.json")):
        with open(json_file) as f:
            data = json.load(f)
            all_results.extend(data['results'])
    
    df = pd.json_normalize(all_results)
    return df

def print_model_comparison(df: pd.DataFrame):
    """Compare performance across models."""
    print("\n" + "="*70)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("="*70)
    
    successful = df[df['status'] == 'success']
    
    if len(successful) > 0:
        summary = successful.groupby('model_name').agg({
            'latency_seconds.mean': ['mean', 'std'],
            'tokens_per_sec.mean': ['mean', 'std'],
        }).round(2)
        
        print("\n" + summary.to_string())
    
    print("="*70 + "\n")

if __name__ == "__main__":
    df = load_results()
    print_model_comparison(df)
