#!/usr/bin/env python3
"""
View CSV Results

Display the generated CSV comparison results in a readable format.
"""

import pandas as pd
import os
from datetime import datetime


def view_csv_results():
    """Display CSV results in a readable format"""
    print("📊 CSV Comparison Results Viewer")
    print("=" * 50)
    
    # Find CSV files
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'ocr_comparison' in f]
    
    if not csv_files:
        print("❌ No CSV comparison files found")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"   • {file}")
    
    print("\n" + "="*50)
    
    # Load and display summary
    summary_files = [f for f in csv_files if 'summary' in f]
    if summary_files:
        print("📈 SUMMARY STATISTICS:")
        print("-" * 30)
        df_summary = pd.read_csv(summary_files[0])
        for _, row in df_summary.iterrows():
            metric = row['metric']
            value = row['value']
            if 'Time' in metric:
                print(f"{metric}: {value:.3f}s")
            elif 'Confidence' in metric:
                print(f"{metric}: {value:.3f}")
            elif 'Speedup' in metric:
                print(f"{metric}: {value:.1f}x")
            elif 'Rate' in metric:
                print(f"{metric}: {value:.1f}%")
            else:
                print(f"{metric}: {value}")
    
    print("\n" + "="*50)
    
    # Load and display performance comparison
    perf_files = [f for f in csv_files if 'performance' in f]
    if perf_files:
        print("⚡ PERFORMANCE COMPARISON:")
        print("-" * 30)
        df_perf = pd.read_csv(perf_files[0])
        
        # Show top 5 fastest custom model results
        print("🏆 Top 5 Fastest Custom Model Results:")
        df_fastest = df_perf.nsmallest(5, 'custom_time_ms')
        for _, row in df_fastest.iterrows():
            print(f"   {row['image_name']}: {row['custom_time_ms']:.1f}ms "
                  f"(vs EasyOCR Std: {row['custom_speedup_vs_std']:.1f}x faster)")
        
        print(f"\n📊 Performance Summary:")
        print(f"   Custom Model: {df_perf['custom_time_ms'].mean():.1f}ms avg")
        print(f"   EasyOCR Std: {df_perf['easyocr_std_time_ms'].mean():.1f}ms avg")
        print(f"   EasyOCR Opt: {df_perf['easyocr_opt_time_ms'].mean():.1f}ms avg")
        print(f"   Custom vs EasyOCR Std: {df_perf['custom_speedup_vs_std'].mean():.1f}x faster")
        print(f"   Custom vs EasyOCR Opt: {df_perf['custom_speedup_vs_opt'].mean():.1f}x faster")
    
    print("\n" + "="*50)
    
    # Load and display detailed comparison
    detailed_files = [f for f in csv_files if 'detailed' in f]
    if detailed_files:
        print("🔍 DETAILED COMPARISON (First 5 images):")
        print("-" * 50)
        df_detailed = pd.read_csv(detailed_files[0])
        
        for i, (_, row) in enumerate(df_detailed.head(5).iterrows()):
            print(f"\n📸 {row['image_name']} ({row['image_size_category']}):")
            print(f"   Custom: '{row['custom_text']}' ({row['custom_confidence']:.3f}) - {row['custom_time_seconds']*1000:.1f}ms")
            print(f"   EasyOCR Std: '{row['easyocr_std_text']}' ({row['easyocr_std_confidence']:.3f}) - {row['easyocr_std_time_seconds']*1000:.1f}ms")
            print(f"   EasyOCR Opt: '{row['easyocr_opt_text']}' ({row['easyocr_opt_confidence']:.3f}) - {row['easyocr_opt_time_seconds']*1000:.1f}ms")
            print(f"   Speedup: {row['speedup_vs_easyocr_std']:.1f}x vs Std, {row['speedup_vs_easyocr_opt']:.1f}x vs Opt")
    
    print(f"\n💡 CSV Files Available:")
    for file in csv_files:
        print(f"   • {file}")
    
    print(f"\n📋 You can open these CSV files in Excel, Google Sheets, or any spreadsheet application")
    print(f"   for detailed analysis and visualization.")


if __name__ == "__main__":
    view_csv_results() 