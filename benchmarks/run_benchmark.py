"""
Systematic LLM inference benchmark runner.
Tests multiple models across configurations and logs all metrics.
"""

import json
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import numpy as np

from src.inference_cli import InferenceCLIWrapper


class BenchmarkRunner:
    def __init__(self, config_path: str, output_dir: str = "results/raw_results"):
        """Initialize benchmark runner with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def run_all_experiments(self) -> List[Dict]:
        """Run full benchmark suite across all models and configurations."""
        
        self.start_time = datetime.now()
        print("\n" + "="*70)
        print("🚀 LLM INFERENCE BENCHMARK SUITE")
        print("="*70)
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Config: {self.config['models'].__len__()} models, "
              f"{len(self.config['prompts'])} prompts, "
              f"{len(self.config['benchmark_settings']['max_tokens'])} token counts")
        print("="*70 + "\n")
        
        # Iterate through all combinations
        total_experiments = (
            len(self.config['models']) *
            len(self.config['prompts']) *
            len(self.config['benchmark_settings']['max_tokens']) *
            len(self.config['benchmark_settings']['temperatures'])
        )
        
        experiment_count = 0
        
        with tqdm(total=total_experiments, desc="Overall progress") as pbar:
            for model_config in self.config['models']:
                model_id = model_config['id']
                model_name = model_config['name']
                
                print(f"\n{'─'*70}")
                print(f"📦 Model: {model_name}")
                print(f"   ID: {model_id}")
                print(f"{'─'*70}")
                
                wrapper = InferenceCLIWrapper(model_id)
                
                for prompt_idx, prompt in enumerate(self.config['prompts']):
                    for max_tokens in self.config['benchmark_settings']['max_tokens']:
                        for temperature in self.config['benchmark_settings']['temperatures']:
                            
                            # Run multiple iterations
                            latencies = []
                            tps_values = []
                            success = True
                            error_msg = None
                            
                            for run_idx in range(self.config['benchmark_settings']['num_runs']):
                                try:
                                    metrics = wrapper.generate_with_metrics(
                                        prompt=prompt,
                                        max_tokens=max_tokens
                                    )
                                    
                                    if metrics['ok']:
                                        latencies.append(metrics['latency_seconds'])
                                        tps_values.append(metrics['tokens_per_sec'])
                                    else:
                                        success = False
                                        error_msg = metrics.get('error', 'Unknown error')
                                        break
                                
                                except Exception as e:
                                    success = False
                                    error_msg = str(e)
                                    break
                            
                            # Log result
                            if success and latencies:
                                result = {
                                    'model_name': model_name,
                                    'model_id': model_id,
                                    'prompt_idx': prompt_idx,
                                    'prompt_preview': prompt[:50],
                                    'max_tokens': max_tokens,
                                    'temperature': temperature,
                                    'num_runs': len(latencies),
                                    'latency_seconds': {
                                        'mean': float(np.mean(latencies)),
                                        'std': float(np.std(latencies)),
                                        'min': float(np.min(latencies)),
                                        'max': float(np.max(latencies)),
                                    },
                                    'tokens_per_sec': {
                                        'mean': float(np.mean(tps_values)),
                                        'std': float(np.std(tps_values)),
                                        'min': float(np.min(tps_values)),
                                        'max': float(np.max(tps_values)),
                                    },
                                    'timestamp': datetime.now().isoformat(),
                                    'status': 'success',
                                }
                                
                                self.results.append(result)
                                
                                # Print summary
                                print(f"  ✅ Tokens={max_tokens:3d}, Temp={temperature:.1f}: "
                                      f"Latency={result['latency_seconds']['mean']:5.2f}s ± "
                                      f"{result['latency_seconds']['std']:4.2f}s, "
                                      f"TPS={result['tokens_per_sec']['mean']:5.1f}")
                            
                            else:
                                # Log failure
                                result = {
                                    'model_name': model_name,
                                    'model_id': model_id,
                                    'prompt_idx': prompt_idx,
                                    'max_tokens': max_tokens,
                                    'temperature': temperature,
                                    'status': 'failed',
                                    'error': error_msg,
                                    'timestamp': datetime.now().isoformat(),
                                }
                                self.results.append(result)
                                print(f"  ❌ Tokens={max_tokens:3d}, Temp={temperature:.1f}: "
                                      f"FAILED - {error_msg}")
                            
                            experiment_count += 1
                            pbar.update(1)
        
        self.end_time = datetime.now()
        self._save_results()
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON file."""
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"benchmark_{timestamp}.json"
        
        data = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': (self.end_time - self.start_time).total_seconds(),
                'total_experiments': len(self.results),
                'successful': sum(1 for r in self.results if r['status'] == 'success'),
                'failed': sum(1 for r in self.results if r['status'] == 'failed'),
            },
            'results': self.results,
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Results saved: {output_file}")
        return output_file
    
    def _print_summary(self):
        """Print summary statistics."""
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        print("\n" + "="*70)
        print("📊 BENCHMARK SUMMARY")
        print("="*70)
        print(f"Total experiments: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Duration: {(self.end_time - self.start_time).total_seconds():.1f}s")
        
        if successful:
            all_latencies = []
            all_tps = []
            
            for r in successful:
                all_latencies.append(r['latency_seconds']['mean'])
                all_tps.append(r['tokens_per_sec']['mean'])
            
            print(f"\n⏱️  Latency (across all runs):")
            print(f"   Mean: {np.mean(all_latencies):.2f}s")
            print(f"   Min: {np.min(all_latencies):.2f}s")
            print(f"   Max: {np.max(all_latencies):.2f}s")
            
            print(f"\n⚡ Throughput (tokens/sec):")
            print(f"   Mean: {np.mean(all_tps):.1f} tokens/sec")
            print(f"   Min: {np.min(all_tps):.1f} tokens/sec")
            print(f"   Max: {np.max(all_tps):.1f} tokens/sec")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    # Run benchmark suite
    runner = BenchmarkRunner("config/models_config.yaml")
    results = runner.run_all_experiments()
