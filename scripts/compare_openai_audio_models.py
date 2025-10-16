#!/usr/bin/env python3
"""
Compare OpenAI Audio Models: Whisper vs GPT-4o vs GPT-4o-mini

This script tests three OpenAI audio transcription models on representative sample files.

Models tested:
1. whisper-1 - Current baseline (Whisper Large v3)
2. gpt-4o-audio-preview - Multimodal GPT-4o with native audio understanding
3. gpt-4o-mini-audio-preview - Efficient multimodal model

Usage:
    python scripts/compare_openai_audio_models.py
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
TEST_FILES = [
    "+919742536994_3_converted.mp3",                     # Ultra-short: 10 sec
    "+919742536994_4_converted.mp3",                     # Short: 2.5 min
    "+917259326110_Fakkirswami S V_converted.mp3",       # Medium: 9.6 min
    "+917259326110_Fakkirswami S V_2_converted.mp3",     # Long: 16.2 min
    "GLPS AMBEDKAR NAGAR GUTTAL.mp3",                    # Very long: 20.7 min
]

MODELS = [
    "whisper-1",
    "gpt-4o-audio-preview",
    "gpt-4o-mini-audio-preview"
]

# Paths
AUDIO_DIR = Path(__file__).parent.parent / "files" / "Haveri_Audios"
OUTPUT_DIR = Path(__file__).parent.parent / "files" / "transcriptions" / "openai_model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def transcribe_with_model(audio_path: Path, model: str) -> dict:
    """
    Transcribe audio file using specified OpenAI model.

    Args:
        audio_path: Path to audio file
        model: Model name (whisper-1, gpt-4o-audio-preview, etc.)

    Returns:
        Dictionary containing transcription result and metadata
    """
    start_time = time.time()

    try:
        with open(audio_path, 'rb') as audio_file:
            # Call OpenAI Audio API
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language="kn",  # Kannada
                response_format="verbose_json"
            )

        processing_time = time.time() - start_time

        # Extract response data
        result = {
            'success': True,
            'model': model,
            'file': audio_path.name,
            'text': response.text,
            'language': response.language,
            'duration': response.duration,
            'segments': [{
                'id': seg.id,
                'start': seg.start,
                'end': seg.end,
                'text': seg.text,
                'avg_logprob': seg.avg_logprob,
                'no_speech_prob': seg.no_speech_prob
            } for seg in response.segments] if hasattr(response, 'segments') else [],
            'processing_time_seconds': processing_time,
            'processed_at': datetime.now().isoformat()
        }

        return result

    except Exception as e:
        processing_time = time.time() - start_time
        return {
            'success': False,
            'model': model,
            'file': audio_path.name,
            'error': str(e),
            'processing_time_seconds': processing_time,
            'processed_at': datetime.now().isoformat()
        }


def main():
    """Main execution function."""
    print("=" * 80)
    print("OpenAI Audio Models Comparison")
    print("=" * 80)
    print(f"Test files: {len(TEST_FILES)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    results = []

    for filename in TEST_FILES:
        audio_path = AUDIO_DIR / filename

        if not audio_path.exists():
            print(f"⚠️  File not found: {filename} (expected at {audio_path})")
            continue

        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print(f"{'='*80}")

        file_results = {'filename': filename, 'models': {}}

        for model in MODELS:
            print(f"\n  Testing model: {model}...")

            result = transcribe_with_model(audio_path, model)
            file_results['models'][model] = result

            if result['success']:
                duration = result.get('duration', 0)
                processing_time = result['processing_time_seconds']
                segments = len(result.get('segments', []))

                print(f"    ✅ Success")
                print(f"       Duration: {duration:.1f}s")
                print(f"       Processing time: {processing_time:.1f}s ({processing_time/max(duration, 1):.2f}x realtime)")
                print(f"       Segments: {segments}")
                print(f"       Text preview: {result['text'][:100]}...")
            else:
                print(f"    ❌ Failed: {result['error']}")

            # Save individual result
            output_file = OUTPUT_DIR / f"{Path(filename).stem}_{model.replace('-', '_')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Small delay to avoid rate limits
            time.sleep(1)

        results.append(file_results)

    # Save comparison summary
    summary_file = OUTPUT_DIR / f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"✅ Comparison test complete!")
    print(f"{'='*80}")
    print(f"Summary saved to: {summary_file}\n")

    # Generate comparison metrics
    comparison_data = []

    for file_result in results:
        filename = file_result['filename']

        for model, result in file_result['models'].items():
            if result['success']:
                comparison_data.append({
                    'filename': filename,
                    'model': model,
                    'duration_sec': result.get('duration', 0),
                    'processing_time_sec': result['processing_time_seconds'],
                    'realtime_factor': result['processing_time_seconds'] / max(result.get('duration', 1), 1),
                    'segment_count': len(result.get('segments', [])),
                    'avg_logprob': sum(s['avg_logprob'] for s in result.get('segments', [])) / max(len(result.get('segments', [])), 1) if result.get('segments') else 0,
                    'avg_no_speech_prob': sum(s['no_speech_prob'] for s in result.get('segments', [])) / max(len(result.get('segments', [])), 1) if result.get('segments') else 0,
                    'text_length': len(result['text'])
                })

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        # Display comparison metrics
        print("\n" + "="*80)
        print("COMPARISON METRICS")
        print("="*80)

        print("\nProcessing Speed (lower is faster):")
        print(df.groupby('model')['realtime_factor'].mean().sort_values())

        print("\nAverage Segments per File:")
        print(df.groupby('model')['segment_count'].mean())

        print("\nAverage Confidence (avg_logprob, higher is better):")
        print(df.groupby('model')['avg_logprob'].mean())

        print("\nAverage No-Speech Probability (lower suggests more actual speech):")
        print(df.groupby('model')['avg_no_speech_prob'].mean())

        print("\nAverage Text Length:")
        print(df.groupby('model')['text_length'].mean())

        # Save detailed comparison
        csv_file = OUTPUT_DIR / f"comparison_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nDetailed metrics saved to: {csv_file}")

        # Display side-by-side transcriptions
        print("\n" + "="*80)
        print("SIDE-BY-SIDE TRANSCRIPTION COMPARISON")
        print("="*80)

        for file_result in results:
            filename = file_result['filename']
            print(f"\n\n{'='*80}")
            print(f"File: {filename}")
            print(f"{'='*80}")

            for model in MODELS:
                result = file_result['models'].get(model, {})
                if result.get('success'):
                    print(f"\n{model}:")
                    print("-" * 80)
                    print(result['text'][:500])  # First 500 chars
                    if len(result['text']) > 500:
                        print("[...]")
                else:
                    print(f"\n{model}: FAILED - {result.get('error', 'Unknown error')}")

    else:
        print("\n⚠️  No successful transcriptions to compare")


if __name__ == "__main__":
    main()
