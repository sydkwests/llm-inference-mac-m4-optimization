"""
Production-grade analysis for LLM inference benchmarks.
Loads all results, computes statistics, generates publication-quality figures.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set publication-quality matplotlib defaults
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BenchmarkAnalyzer:
    def __init__(self, results_dir: str = "results/raw_results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path("results/figures")
        self.analysis_dir = Path("results/analysis")
        
        # Create output directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all results FIRST
        self.df = self._load_results()
        
        # THEN filter successful results
        self.successful_df = self.df[self.df['status'] == 'success'].copy()
        
        # Print summary
        print(f"✅ Loaded {len(self.df)} total experiments")
        print(f"   ✅ {len(self.successful_df)} successful")
        print(f"   ❌ {len(self.df) - len(self.successful_df)} failed\n")
    
    def _load_results(self) -> pd.DataFrame:
        """Load all benchmark JSON files into a single DataFrame."""
        all_results = []
        
        for json_file in sorted(self.results_dir.glob("benchmark_*.json")):
            with open(json_file) as f:
                data = json.load(f)
                all_results.extend(data['results'])
        
        if len(all_results) == 0:
            print("⚠️  No results found! Make sure you've run the benchmark.")
            return pd.DataFrame()
        
        df = pd.json_normalize(all_results)
        return df
    
    def compute_summary_statistics(self) -> pd.DataFrame:
        """Compute mean/std across all models."""
        if len(self.successful_df) == 0:
            print("No successful results to summarize")
            return None
        
        summary = self.successful_df.groupby('model_name').agg({
            'latency_seconds.mean': ['mean', 'std', 'min', 'max'],
            'tokens_per_sec.mean': ['mean', 'std', 'min', 'max'],
        }).round(3)
        
        print("\n" + "="*80)
        print("📊 SUMMARY STATISTICS BY MODEL")
        print("="*80)
        print(summary)
        print("="*80 + "\n")
        
        # Save to CSV
        summary.to_csv(self.analysis_dir / "summary_stats.csv")
        print(f"✅ Saved: {self.analysis_dir / 'summary_stats.csv'}\n")
        
        return summary
    
    def compute_token_scaling(self) -> pd.DataFrame:
        """Analyze how latency scales with token length."""
        if len(self.successful_df) == 0:
            return None
        
        token_scaling = self.successful_df.groupby(['model_name', 'max_tokens']).agg({
            'latency_seconds.mean': ['mean', 'std'],
            'tokens_per_sec.mean': ['mean', 'std'],
        }).round(3)
        
        print("\n" + "="*80)
        print("📈 TOKEN SCALING ANALYSIS")
        print("="*80)
        print(token_scaling)
        print("="*80 + "\n")
        
        token_scaling.to_csv(self.analysis_dir / "token_scaling.csv")
        return token_scaling
    
    def create_figure_latency_by_model(self):
        """Figure 1: Latency comparison across models (boxplot)."""
        if len(self.successful_df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sns.boxplot(
            data=self.successful_df,
            x='model_name',
            y='latency_seconds.mean',
            ax=ax,
            palette='Set2',
            width=0.6
        )
        
        ax.set_title('Inference Latency by Model', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Latency (seconds)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.figures_dir / "01_latency_by_model.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {path}")
    
    def create_figure_throughput_by_model(self):
        """Figure 2: Throughput comparison across models."""
        if len(self.successful_df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sns.boxplot(
            data=self.successful_df,
            x='model_name',
            y='tokens_per_sec.mean',
            ax=ax,
            palette='Set2',
            width=0.6
        )
        
        ax.set_title('Inference Throughput by Model', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Tokens/Second', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.figures_dir / "02_throughput_by_model.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {path}")
    
    def create_figure_latency_vs_tokens(self):
        """Figure 3: Latency vs output length (line plot per model)."""
        if len(self.successful_df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Group by model and token length, get mean latency
        token_data = self.successful_df.groupby(['model_name', 'max_tokens'])['latency_seconds.mean'].mean().reset_index()
        
        if len(token_data) == 0:
            return
        
        for model in token_data['model_name'].unique():
            data = token_data[token_data['model_name'] == model]
            ax.plot(
                data['max_tokens'],
                data['latency_seconds.mean'],
                marker='o',
                linewidth=2.5,
                markersize=8,
                label=model
            )
        
        ax.set_title('Latency Scaling with Output Length', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Max Tokens', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency (seconds)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.figures_dir / "03_latency_vs_tokens.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {path}")
    
    def create_figure_temperature_effect(self):
        """Figure 4: Temperature effect on latency (should be minimal)."""
        if len(self.successful_df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sns.boxplot(
            data=self.successful_df,
            x='temperature',
            y='latency_seconds.mean',
            hue='model_name',
            ax=ax,
            palette='Set2'
        )
        
        ax.set_title('Effect of Temperature on Latency', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Latency (seconds)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Temperature', fontsize=14, fontweight='bold')
        ax.legend(title='Model', fontsize=11, title_fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.figures_dir / "04_temperature_effect.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {path}")
    
    def create_figure_latency_distribution(self):
        """Figure 5: Distribution of latencies (histogram)."""
        if len(self.successful_df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for model in self.successful_df['model_name'].unique():
            data = self.successful_df[self.successful_df['model_name'] == model]['latency_seconds.mean']
            ax.hist(data, bins=10, alpha=0.6, label=model)
        
        ax.set_title('Distribution of Inference Latencies', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Latency (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.figures_dir / "05_latency_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {path}")
    
    def generate_all_figures(self):
        """Generate all publication-quality figures."""
        print("\n🎨 Generating publication-quality figures...")
        self.create_figure_latency_by_model()
        self.create_figure_throughput_by_model()
        self.create_figure_latency_vs_tokens()
        self.create_figure_temperature_effect()
        self.create_figure_latency_distribution()
        print("✅ All figures generated!\n")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*80)
        print("🚀 RUNNING COMPREHENSIVE BENCHMARK ANALYSIS")
        print("="*80 + "\n")
        
        self.compute_summary_statistics()
        self.compute_token_scaling()
        self.generate_all_figures()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
        print(f"📁 Figures: {self.figures_dir}/")
        print(f"📊 CSV files: {self.analysis_dir}/")
        print("="*80 + "\n")


if __name__ == "__main__":
    analyzer = BenchmarkAnalyzer()
    analyzer.run_full_analysis()
