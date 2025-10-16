#!/usr/bin/env python3
"""
Transcription Evaluation Script

Computes Word Error Rate (WER) and Character Error Rate (CER) between two sets of
transcriptions. Designed to be generic - accepts any two directories as "golden"
(reference) and "generated" (hypothesis) for easy swapping when ground truth becomes
available.

Usage:
    # Compare Azure vs Whisper transcriptions
    python scripts/evaluate_transcriptions.py \
        --golden files/transcriptions/azure_batch \
        --generated files/transcriptions/batch_whisper_gpt4o \
        --script both \
        --output files/evaluations/azure_vs_whisper.json

    # Evaluate only Kannada script
    python scripts/evaluate_transcriptions.py \
        --golden files/transcriptions/azure_batch \
        --generated files/transcriptions/batch_whisper_gpt4o \
        --script kannada \
        --output files/evaluations/kannada_only.json

    # Evaluate only Roman transliteration
    python scripts/evaluate_transcriptions.py \
        --golden files/transcriptions/azure_batch \
        --generated files/transcriptions/batch_whisper_gpt4o \
        --script roman \
        --output files/evaluations/roman_only.json

Args:
    --golden: Path to directory with reference (golden) transcriptions
    --generated: Path to directory with hypothesis (generated) transcriptions
    --script: Which script to evaluate (kannada, roman, both) - default: both
    --output: Path to save evaluation results JSON
    --normalize: Apply text normalization (lowercase, punctuation removal) - default: True
    --inventory: Path to common files inventory JSON (optional)

Metrics:
    - WER (Word Error Rate): (Insertions + Deletions + Substitutions) / Total Words
    - CER (Character Error Rate): (Insertions + Deletions + Substitutions) / Total Characters
    - Computed separately for Kannada native script and Roman transliteration
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer


def normalize_text(text: str, script: str = "kannada") -> str:
    """
    Normalize text for fair comparison.

    Args:
        text: Input text to normalize
        script: Script type ('kannada' or 'roman')

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Unicode normalization (NFC form - canonical composition)
    text = unicodedata.normalize('NFC', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    if script == "roman":
        # Lowercase for Roman script
        text = text.lower()

        # Remove common punctuation
        text = re.sub(r'[.,!?;:\-\[\](){}\"\'`]', '', text)

        # Remove ellipsis patterns
        text = re.sub(r'\.{2,}', '', text)
    else:
        # For Kannada, just remove common punctuation
        # Keep Kannada punctuation marks like । (danda)
        text = re.sub(r'[.,!?;:\-\[\](){}\"\'`]', '', text)
        text = re.sub(r'\.{2,}', '', text)

    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def extract_text_from_azure(data: Dict) -> Tuple[str, str]:
    """
    Extract Kannada and Roman text from Azure transcription format.

    Args:
        data: Azure transcription JSON data

    Returns:
        Tuple of (kannada_text, roman_text)
    """
    kannada = data.get('text', '')
    roman = data.get('romanized_text', '')

    return kannada, roman


def extract_text_from_whisper(data: Dict) -> Tuple[str, str]:
    """
    Extract Kannada and Roman text from Whisper transcription format.

    Args:
        data: Whisper transcription JSON data

    Returns:
        Tuple of (kannada_text, roman_text)
    """
    transcription = data.get('transcription', {})
    kannada = transcription.get('text_kannada', '')
    roman = transcription.get('text_romanized', '')

    return kannada, roman


def detect_format(data: Dict) -> str:
    """
    Detect transcription format (azure or whisper).

    Args:
        data: Transcription JSON data

    Returns:
        Format string ('azure' or 'whisper')
    """
    # Azure has 'text' and 'romanized_text' at top level
    if 'text' in data and 'romanized_text' in data:
        return 'azure'

    # Whisper has nested 'transcription' object
    if 'transcription' in data and 'text_kannada' in data.get('transcription', {}):
        return 'whisper'

    raise ValueError("Unknown transcription format - expected Azure or Whisper format")


def load_transcription(file_path: Path) -> Tuple[str, str, str]:
    """
    Load transcription file and extract texts.

    Args:
        file_path: Path to transcription JSON file

    Returns:
        Tuple of (format, kannada_text, roman_text)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fmt = detect_format(data)

    if fmt == 'azure':
        kannada, roman = extract_text_from_azure(data)
    else:  # whisper
        kannada, roman = extract_text_from_whisper(data)

    return fmt, kannada, roman


def find_matching_files(
    golden_dir: Path,
    generated_dir: Path,
    inventory_path: Optional[Path] = None
) -> List[Tuple[Path, Path]]:
    """
    Find matching transcription files between golden and generated directories.

    Uses inventory file if provided, otherwise does fuzzy matching based on filenames.

    Args:
        golden_dir: Directory with golden transcriptions
        generated_dir: Directory with generated transcriptions
        inventory_path: Optional path to common_files_inventory.json

    Returns:
        List of (golden_file, generated_file) path tuples
    """
    matches = []

    if inventory_path and inventory_path.exists():
        # Use inventory file for matching
        with open(inventory_path, 'r', encoding='utf-8') as f:
            inventory = json.load(f)

        # Determine which set is Azure and which is Whisper
        # by checking a sample file from each directory
        # Filter for transcription files only (not batch_summary files)
        golden_files = [
            f for f in golden_dir.glob('*.json')
            if not f.name.startswith('batch_summary')
            and (f.name.endswith('_azure_full.json') or f.name.endswith('.json'))
        ]
        generated_files = [
            f for f in generated_dir.glob('*.json')
            if not f.name.startswith('batch_summary')
            and (f.name.endswith('_converted.json') or f.name.endswith('.json'))
        ]

        if not golden_files or not generated_files:
            raise ValueError("No JSON files found in golden or generated directories")

        # Try to find a valid transcription file
        golden_format = None
        for gf in golden_files[:5]:  # Try first 5 files
            try:
                with open(gf, 'r', encoding='utf-8') as f:
                    golden_sample = json.load(f)
                golden_format = detect_format(golden_sample)
                break
            except:
                continue

        if not golden_format:
            raise ValueError("Could not detect format from golden directory files")

        generated_format = None
        for gf in generated_files[:5]:  # Try first 5 files
            try:
                with open(gf, 'r', encoding='utf-8') as f:
                    generated_sample = json.load(f)
                generated_format = detect_format(generated_sample)
                break
            except:
                continue

        if not generated_format:
            raise ValueError("Could not detect format from generated directory files")

        print(f"Golden directory format: {golden_format}")
        print(f"Generated directory format: {generated_format}")

        # Match files based on inventory
        for item in inventory['common_files']:
            if golden_format == 'azure':
                golden_file = golden_dir / item['azure_file']
                generated_file = generated_dir / item['whisper_file']
            else:
                golden_file = golden_dir / item['whisper_file']
                generated_file = generated_dir / item['azure_file']

            if golden_file.exists() and generated_file.exists():
                matches.append((golden_file, generated_file))
            else:
                print(f"⚠️  Warning: Missing files for {item['base_name']}")
    else:
        # Fuzzy matching based on filenames (fallback)
        print("⚠️  No inventory file provided, using fuzzy filename matching")

        golden_files = {f.stem: f for f in golden_dir.glob('*.json')}
        generated_files = {f.stem: f for f in generated_dir.glob('*.json')}

        # Try exact stem matches
        for stem, golden_file in golden_files.items():
            if stem in generated_files:
                matches.append((golden_file, generated_files[stem]))

        # Try fuzzy matching on base names (remove suffixes like _azure_full, _converted)
        if not matches:
            for g_stem, golden_file in golden_files.items():
                # Clean golden stem
                g_base = re.sub(r'_azure_full|_converted|_whisper', '', g_stem)
                g_base = re.sub(r'[_\s]+', '_', g_base)

                for gen_stem, generated_file in generated_files.items():
                    # Clean generated stem
                    gen_base = re.sub(r'_azure_full|_converted|_whisper', '', gen_stem)
                    gen_base = re.sub(r'[_\s]+', '_', gen_base)

                    if g_base.lower() == gen_base.lower():
                        matches.append((golden_file, generated_file))
                        break

    return matches


def compute_metrics(
    reference: str,
    hypothesis: str,
    script: str,
    normalize: bool = True
) -> Dict:
    """
    Compute WER and CER metrics.

    Args:
        reference: Reference (golden) text
        hypothesis: Hypothesis (generated) text
        script: Script type ('kannada' or 'roman')
        normalize: Whether to apply text normalization

    Returns:
        Dictionary with WER, CER, and component metrics
    """
    # Normalize if requested
    if normalize:
        reference = normalize_text(reference, script)
        hypothesis = normalize_text(hypothesis, script)

    # Handle empty cases
    if not reference and not hypothesis:
        return {
            'wer': 0.0,
            'cer': 0.0,
            'reference_words': 0,
            'hypothesis_words': 0,
            'reference_chars': 0,
            'hypothesis_chars': 0
        }

    if not reference:
        return {
            'wer': float('inf'),
            'cer': float('inf'),
            'reference_words': 0,
            'hypothesis_words': len(hypothesis.split()),
            'reference_chars': 0,
            'hypothesis_chars': len(hypothesis)
        }

    if not hypothesis:
        return {
            'wer': 1.0,
            'cer': 1.0,
            'reference_words': len(reference.split()),
            'hypothesis_words': 0,
            'reference_chars': len(reference),
            'hypothesis_chars': 0
        }

    # Compute WER (Word Error Rate)
    wer = jiwer.wer(reference, hypothesis)

    # Compute CER (Character Error Rate)
    cer = jiwer.cer(reference, hypothesis)

    return {
        'wer': wer,
        'cer': cer,
        'reference_words': len(reference.split()),
        'hypothesis_words': len(hypothesis.split()),
        'reference_chars': len(reference),
        'hypothesis_chars': len(hypothesis),
        'reference_text': reference,
        'hypothesis_text': hypothesis
    }


def evaluate_file_pair(
    golden_file: Path,
    generated_file: Path,
    script: str,
    normalize: bool = True
) -> Dict:
    """
    Evaluate a single pair of transcription files at file level.

    Args:
        golden_file: Path to golden transcription
        generated_file: Path to generated transcription
        script: Which script to evaluate ('kannada', 'roman', 'both')
        normalize: Whether to normalize text

    Returns:
        Dictionary with evaluation results
    """
    # Load transcriptions
    golden_fmt, golden_kannada, golden_roman = load_transcription(golden_file)
    generated_fmt, generated_kannada, generated_roman = load_transcription(generated_file)

    result = {
        'base_name': golden_file.stem,
        'golden_file': golden_file.name,
        'generated_file': generated_file.name,
        'golden_format': golden_fmt,
        'generated_format': generated_fmt
    }

    # Evaluate Kannada script
    if script in ['kannada', 'both']:
        kannada_metrics = compute_metrics(
            golden_kannada,
            generated_kannada,
            'kannada',
            normalize
        )
        result['kannada'] = kannada_metrics

    # Evaluate Roman transliteration
    if script in ['roman', 'both']:
        roman_metrics = compute_metrics(
            golden_roman,
            generated_roman,
            'roman',
            normalize
        )
        result['roman'] = roman_metrics

    return result


def compute_aggregate_stats(results: List[Dict], script_key: str) -> Dict:
    """
    Compute aggregate statistics across all files.

    Args:
        results: List of per-file evaluation results
        script_key: Script key ('kannada' or 'roman')

    Returns:
        Dictionary with aggregate statistics
    """
    import statistics

    # Extract WER/CER values from file results
    wers = [r[script_key]['wer'] for r in results if script_key in r and r[script_key]['wer'] != float('inf')]
    cers = [r[script_key]['cer'] for r in results if script_key in r and r[script_key]['cer'] != float('inf')]

    if not wers or not cers:
        return {
            'wer_mean': None,
            'wer_median': None,
            'wer_min': None,
            'wer_max': None,
            'cer_mean': None,
            'cer_median': None,
            'cer_min': None,
            'cer_max': None,
            'total_files': len(results)
        }

    return {
        'wer_mean': statistics.mean(wers),
        'wer_median': statistics.median(wers),
        'wer_min': min(wers),
        'wer_max': max(wers),
        'wer_stdev': statistics.stdev(wers) if len(wers) > 1 else 0.0,
        'cer_mean': statistics.mean(cers),
        'cer_median': statistics.median(cers),
        'cer_min': min(cers),
        'cer_max': max(cers),
        'cer_stdev': statistics.stdev(cers) if len(cers) > 1 else 0.0,
        'total_files': len(results),
        'valid_files': len(wers)
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Evaluate transcriptions using WER and CER metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--golden',
        type=Path,
        required=True,
        help='Path to directory with reference (golden) transcriptions'
    )

    parser.add_argument(
        '--generated',
        type=Path,
        required=True,
        help='Path to directory with hypothesis (generated) transcriptions'
    )

    parser.add_argument(
        '--script',
        type=str,
        choices=['kannada', 'roman', 'both'],
        default='both',
        help='Which script to evaluate (default: both)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to save evaluation results JSON'
    )

    parser.add_argument(
        '--inventory',
        type=Path,
        help='Path to common_files_inventory.json (optional, for better file matching)'
    )

    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable text normalization (not recommended)'
    )

    parser.add_argument(
        '--csv',
        type=Path,
        help='Optional: Export results to CSV file for easy inspection (e.g., results.csv)'
    )

    args = parser.parse_args()

    # Validate directories
    if not args.golden.is_dir():
        print(f"❌ Error: Golden directory not found: {args.golden}")
        sys.exit(1)

    if not args.generated.is_dir():
        print(f"❌ Error: Generated directory not found: {args.generated}")
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Print header
    print("=" * 80)
    print("TRANSCRIPTION EVALUATION")
    print("=" * 80)
    print(f"Golden:     {args.golden}")
    print(f"Generated:  {args.generated}")
    print(f"Script:     {args.script}")
    print(f"Normalize:  {not args.no_normalize}")
    print("=" * 80)

    # Find matching files
    print("\nFinding matching transcription files...")
    matches = find_matching_files(args.golden, args.generated, args.inventory)

    if not matches:
        print("❌ No matching files found between golden and generated directories")
        sys.exit(1)

    print(f"✓ Found {len(matches)} matching file pairs\n")

    # Evaluate each file pair
    results = []

    for idx, (golden_file, generated_file) in enumerate(matches, 1):
        print(f"[{idx}/{len(matches)}] Evaluating {golden_file.stem}...", end=' ')

        try:
            result = evaluate_file_pair(
                golden_file,
                generated_file,
                args.script,
                normalize=not args.no_normalize
            )
            results.append(result)

            # Print quick summary
            if args.script == 'both':
                print(f"✓ (KN WER: {result['kannada']['wer']:.3f}, Roman WER: {result['roman']['wer']:.3f})")
            elif args.script == 'kannada':
                print(f"✓ (WER: {result['kannada']['wer']:.3f})")
            else:
                print(f"✓ (WER: {result['roman']['wer']:.3f})")

        except Exception as e:
            print(f"❌ Failed: {e}")
            continue

    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    summary = {
        'evaluation_metadata': {
            'golden_dir': str(args.golden),
            'generated_dir': str(args.generated),
            'script': args.script,
            'normalized': not args.no_normalize,
            'total_files': len(results),
            'evaluated_at': datetime.now().isoformat()
        },
        'per_file_results': results
    }

    # Add aggregate stats
    if args.script in ['kannada', 'both']:
        kannada_stats = compute_aggregate_stats(results, 'kannada')
        summary['kannada_aggregate'] = kannada_stats

        print("\nKannada Script Metrics:")
        print(f"  WER - Mean: {kannada_stats['wer_mean']:.3f}, Median: {kannada_stats['wer_median']:.3f}, "
              f"Range: [{kannada_stats['wer_min']:.3f}, {kannada_stats['wer_max']:.3f}]")
        print(f"  CER - Mean: {kannada_stats['cer_mean']:.3f}, Median: {kannada_stats['cer_median']:.3f}, "
              f"Range: [{kannada_stats['cer_min']:.3f}, {kannada_stats['cer_max']:.3f}]")

    if args.script in ['roman', 'both']:
        roman_stats = compute_aggregate_stats(results, 'roman')
        summary['roman_aggregate'] = roman_stats

        print("\nRoman Transliteration Metrics:")
        print(f"  WER - Mean: {roman_stats['wer_mean']:.3f}, Median: {roman_stats['wer_median']:.3f}, "
              f"Range: [{roman_stats['wer_min']:.3f}, {roman_stats['wer_max']:.3f}]")
        print(f"  CER - Mean: {roman_stats['cer_mean']:.3f}, Median: {roman_stats['cer_median']:.3f}, "
              f"Range: [{roman_stats['cer_min']:.3f}, {roman_stats['cer_max']:.3f}]")

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Evaluation results saved to: {args.output}")

    # Export to CSV if requested
    if args.csv:
        export_to_csv(results, args.csv, args.script)
        print(f"✓ CSV export saved to: {args.csv}")

    print("=" * 80)


def export_to_csv(results: List[Dict], csv_path: Path, script: str) -> None:
    """
    Export file-level evaluation results to CSV for easy inspection.

    Args:
        results: List of per-file evaluation results
        csv_path: Path to save CSV file
        script: Which script was evaluated ('kannada', 'roman', 'both')
    """
    # Create output directory if needed
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if script == 'both':
            fieldnames = [
                'golden_file',
                'generated_file',
                'kannada_wer',
                'kannada_cer',
                'kannada_ref_words',
                'kannada_hyp_words',
                'kannada_ref_chars',
                'kannada_hyp_chars',
                'roman_wer',
                'roman_cer',
                'roman_ref_words',
                'roman_hyp_words',
                'roman_ref_chars',
                'roman_hyp_chars',
                'kannada_reference',
                'kannada_hypothesis',
                'roman_reference',
                'roman_hypothesis'
            ]
        elif script == 'kannada':
            fieldnames = [
                'golden_file',
                'generated_file',
                'wer',
                'cer',
                'reference_words',
                'hypothesis_words',
                'reference_chars',
                'hypothesis_chars',
                'reference_text',
                'hypothesis_text'
            ]
        else:  # roman
            fieldnames = [
                'golden_file',
                'generated_file',
                'wer',
                'cer',
                'reference_words',
                'hypothesis_words',
                'reference_chars',
                'hypothesis_chars',
                'reference_text',
                'hypothesis_text'
            ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            if script == 'both':
                row = {
                    'golden_file': result['golden_file'],
                    'generated_file': result['generated_file'],
                    'kannada_wer': f"{result['kannada']['wer']:.4f}",
                    'kannada_cer': f"{result['kannada']['cer']:.4f}",
                    'kannada_ref_words': result['kannada']['reference_words'],
                    'kannada_hyp_words': result['kannada']['hypothesis_words'],
                    'kannada_ref_chars': result['kannada']['reference_chars'],
                    'kannada_hyp_chars': result['kannada']['hypothesis_chars'],
                    'roman_wer': f"{result['roman']['wer']:.4f}",
                    'roman_cer': f"{result['roman']['cer']:.4f}",
                    'roman_ref_words': result['roman']['reference_words'],
                    'roman_hyp_words': result['roman']['hypothesis_words'],
                    'roman_ref_chars': result['roman']['reference_chars'],
                    'roman_hyp_chars': result['roman']['hypothesis_chars'],
                    'kannada_reference': result['kannada']['reference_text'],
                    'kannada_hypothesis': result['kannada']['hypothesis_text'],
                    'roman_reference': result['roman']['reference_text'],
                    'roman_hypothesis': result['roman']['hypothesis_text']
                }
            elif script == 'kannada':
                row = {
                    'golden_file': result['golden_file'],
                    'generated_file': result['generated_file'],
                    'wer': f"{result['kannada']['wer']:.4f}",
                    'cer': f"{result['kannada']['cer']:.4f}",
                    'reference_words': result['kannada']['reference_words'],
                    'hypothesis_words': result['kannada']['hypothesis_words'],
                    'reference_chars': result['kannada']['reference_chars'],
                    'hypothesis_chars': result['kannada']['hypothesis_chars'],
                    'reference_text': result['kannada']['reference_text'],
                    'hypothesis_text': result['kannada']['hypothesis_text']
                }
            else:  # roman
                row = {
                    'golden_file': result['golden_file'],
                    'generated_file': result['generated_file'],
                    'wer': f"{result['roman']['wer']:.4f}",
                    'cer': f"{result['roman']['cer']:.4f}",
                    'reference_words': result['roman']['reference_words'],
                    'hypothesis_words': result['roman']['hypothesis_words'],
                    'reference_chars': result['roman']['reference_chars'],
                    'hypothesis_chars': result['roman']['hypothesis_chars'],
                    'reference_text': result['roman']['reference_text'],
                    'hypothesis_text': result['roman']['hypothesis_text']
                }

            writer.writerow(row)


if __name__ == "__main__":
    main()
