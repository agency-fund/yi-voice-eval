#!/usr/bin/env python3
"""
Format Batch Transcription Results to CSV

This script converts JSON transcription results from the Whisper + GPT-4o-mini pipeline
into CSV format for spreadsheet analysis.

Usage:
    python scripts/format_results_to_csv.py [input_folder]

Args:
    input_folder: Path to folder containing JSON files (default: files/transcriptions/batch_whisper_gpt4o)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any


def load_json_files(folder_path: Path) -> List[Dict[str, Any]]:
    """
    Load all JSON files from a folder.

    Args:
        folder_path: Path to folder containing JSON files

    Returns:
        List of dictionaries containing parsed JSON data
    """
    json_files = list(folder_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {folder_path}")

    results = []
    errors = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({
                    'filename': json_file.name,
                    'data': data
                })
        except Exception as e:
            error_msg = f"Error loading {json_file.name}: {str(e)}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")

    print(f"\n✅ Successfully loaded {len(results)} files")
    if errors:
        print(f"⚠️  {len(errors)} files failed to load")

    return results


def generate_summary_csv(json_data: List[Dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """
    Generate summary CSV with one row per file.

    Args:
        json_data: List of parsed JSON data
        output_path: Path to save CSV file

    Returns:
        DataFrame containing summary data
    """
    rows = []

    for item in json_data:
        data = item['data']
        metadata = data.get('metadata', {})
        transcription = data.get('transcription', {})
        segments = data.get('segments', [])
        costs = metadata.get('costs_usd', {})
        metrics = metadata.get('metrics', {})

        row = {
            'filename': metadata.get('file', item['filename']),
            'duration_seconds': metadata.get('duration', 0),
            'language': metadata.get('language', 'unknown'),
            'full_kannada_text': transcription.get('text_kannada', ''),
            'full_romanized_text': transcription.get('text_romanized', ''),
            'segment_count': len(segments),
            'total_tokens': metrics.get('total_tokens', 0),
            'whisper_cost_usd': costs.get('whisper_transcription', 0),
            'gpt_cost_usd': costs.get('gpt_transliteration_segments', 0),
            'total_cost_usd': costs.get('total_pipeline', 0),
            'processed_at': metadata.get('processed_at', '')
        }
        rows.append(row)

    # Create DataFrame and sort by filename
    df = pd.DataFrame(rows)
    df = df.sort_values('filename').reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Summary CSV saved to: {output_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")

    # Calculate totals
    total_duration = df['duration_seconds'].sum()
    total_cost = df['total_cost_usd'].sum()
    print(f"\n📊 Summary Statistics:")
    print(f"   Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    print(f"   Total cost: ${total_cost:.4f}")
    if total_duration > 0:
        print(f"   Average cost per hour: ${(total_cost / (total_duration/3600)):.4f}")

    return df


def generate_detailed_csv(json_data: List[Dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """
    Generate detailed CSV with one row per segment.

    Args:
        json_data: List of parsed JSON data
        output_path: Path to save CSV file

    Returns:
        DataFrame containing detailed segment data
    """
    rows = []

    for item in json_data:
        data = item['data']
        metadata = data.get('metadata', {})
        segments = data.get('segments', [])
        filename = metadata.get('file', item['filename'])

        for segment in segments:
            row = {
                'filename': filename,
                'segment_id': segment.get('id', ''),
                'start_time': segment.get('start', 0),
                'end_time': segment.get('end', 0),
                'duration': segment.get('end', 0) - segment.get('start', 0),
                'kannada_text': segment.get('text', ''),
                'romanized_text': segment.get('text_romanized', ''),
                'avg_logprob': segment.get('avg_logprob', None),
                'no_speech_prob': segment.get('no_speech_prob', None)
            }
            rows.append(row)

    # Create DataFrame and sort by filename and segment_id
    df = pd.DataFrame(rows)
    df = df.sort_values(['filename', 'segment_id']).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Detailed CSV saved to: {output_path}")
    print(f"   Rows: {len(df)} segments")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Files covered: {df['filename'].nunique()}")

    return df


def main():
    """Main execution function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_folder = Path(sys.argv[1])
    else:
        # Default to batch_whisper_gpt4o folder
        input_folder = Path(__file__).parent.parent / "files" / "transcriptions" / "batch_whisper_gpt4o"

    # Output folder
    output_folder = Path(__file__).parent.parent / "files" / "reports"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Timestamp for output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=" * 80)
    print(f"Format Batch Transcription Results to CSV")
    print(f"=" * 80)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Timestamp: {timestamp}\n")

    # Check if input folder exists
    if not input_folder.exists():
        print(f"❌ Error: Input folder does not exist: {input_folder}")
        sys.exit(1)

    # Load JSON files
    json_data = load_json_files(input_folder)

    if not json_data:
        print(f"❌ Error: No valid JSON files found in {input_folder}")
        sys.exit(1)

    # Generate summary CSV
    print(f"\n{'=' * 80}")
    print(f"Generating Summary CSV (File-Level)")
    print(f"{'=' * 80}")
    summary_output = output_folder / f"batch_whisper_gpt4o_summary_{timestamp}.csv"
    summary_df = generate_summary_csv(json_data, summary_output)

    # Also save as "latest" version
    summary_latest = output_folder / "batch_whisper_gpt4o_summary_latest.csv"
    summary_df.to_csv(summary_latest, index=False, encoding='utf-8')
    print(f"✅ Also saved as: {summary_latest}")

    # Preview summary (excluding full text columns)
    print(f"\n📋 Summary CSV Preview (first 5 rows):")
    preview_cols = ['filename', 'duration_seconds', 'language', 'segment_count', 'total_cost_usd']
    print(summary_df[preview_cols].head(5).to_string(index=False))

    # Generate detailed CSV
    print(f"\n{'=' * 80}")
    print(f"Generating Detailed CSV (Segment-Level)")
    print(f"{'=' * 80}")
    detailed_output = output_folder / f"batch_whisper_gpt4o_detailed_{timestamp}.csv"
    detailed_df = generate_detailed_csv(json_data, detailed_output)

    # Also save as "latest" version
    detailed_latest = output_folder / "batch_whisper_gpt4o_detailed_latest.csv"
    detailed_df.to_csv(detailed_latest, index=False, encoding='utf-8')
    print(f"✅ Also saved as: {detailed_latest}")

    # Preview detailed (excluding full text columns)
    print(f"\n📋 Detailed CSV Preview (first 5 rows):")
    preview_cols = ['filename', 'segment_id', 'start_time', 'end_time', 'duration']
    print(detailed_df[preview_cols].head(5).to_string(index=False))

    print(f"\n{'=' * 80}")
    print(f"✅ CSV generation complete!")
    print(f"{'=' * 80}")
    print(f"\nGenerated files:")
    print(f"  1. Summary CSV:  {summary_output}")
    print(f"                   {summary_latest}")
    print(f"  2. Detailed CSV: {detailed_output}")
    print(f"                   {detailed_latest}")
    print(f"\nNext steps:")
    print(f"  - Import CSVs into Google Sheets or Excel for analysis")
    print(f"  - Use detailed CSV for time-based analysis and quality checks")
    print(f"  - Use summary CSV for cost analysis and overview statistics")


if __name__ == "__main__":
    main()
