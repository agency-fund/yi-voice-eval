#!/usr/bin/env python3
"""
Azure Batch Transcription Script

This script transcribes audio files from Azure Blob Storage using Azure Speech Service
with speaker diarization and GPT-4o-mini transliteration.

Usage:
    # Process all audio files (all formats)
    python scripts/azure_batch_transcribe.py

    # Process specific file types
    python scripts/azure_batch_transcribe.py --file-types mp3,wav

    # Process specific files by name
    python scripts/azure_batch_transcribe.py --files "file1.mp3,file2.wav"

    # Disable diarization
    python scripts/azure_batch_transcribe.py --no-diarization

    # Disable transliteration
    python scripts/azure_batch_transcribe.py --no-transliteration

    # Process files one at a time with pauses
    python scripts/azure_batch_transcribe.py --file-types mp3 --pause-between-files 5

Args:
    --file-types: Comma-separated list of file extensions (default: all audio types)
    --files: Comma-separated list of specific filenames to process
    --locale: Language locale code (default: kn-IN for Kannada)
    --no-diarization: Disable speaker diarization
    --no-transliteration: Disable Roman transliteration
    --min-speakers: Minimum speaker count for diarization (default: 2)
    --max-speakers: Maximum speaker count for diarization (default: 2)
    --pause-between-files: Seconds to pause between files (default: 0)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from voice_eval.azure_batch_api import AzureBatchTranscription


# Audio file extensions supported by Azure Speech Service
AUDIO_EXTENSIONS = ['mp3', 'mp4', 'wav', 'aac', 'amr', 'ogg', 'flac', 'm4a']


def list_audio_files(
    container_client,
    file_types: Optional[List[str]] = None,
    specific_files: Optional[List[str]] = None
) -> List[str]:
    """
    List audio files from Azure Blob Storage.

    Args:
        container_client: Azure Blob container client
        file_types: List of file extensions to filter (e.g., ['mp3', 'wav'])
        specific_files: List of specific filenames to process

    Returns:
        Sorted list of blob names
    """
    all_blobs = [b.name for b in container_client.list_blobs()]

    # Filter by specific files if provided
    if specific_files:
        matching_blobs = [b for b in all_blobs if b in specific_files]
        print(f"Found {len(matching_blobs)}/{len(specific_files)} specified files")
        return sorted(matching_blobs)

    # Filter by file types
    if file_types:
        extensions = tuple(f'.{ext.lower()}' for ext in file_types)
        audio_blobs = [b for b in all_blobs if b.lower().endswith(extensions)]
    else:
        # Default: all audio files
        extensions = tuple(f'.{ext.lower()}' for ext in AUDIO_EXTENSIONS)
        audio_blobs = [b for b in all_blobs if b.lower().endswith(extensions)]
        # Exclude non-audio files
        audio_blobs = [b for b in audio_blobs if not b.lower().endswith(('.jpeg', '.jpg', '.png', '.pdf'))]

    return sorted(audio_blobs)


def process_files(
    client: AzureBatchTranscription,
    blob_names: List[str],
    output_dir: Path,
    locale: str = "kn-IN",
    enable_diarization: bool = True,
    enable_transliteration: bool = True,
    min_speakers: int = 2,
    max_speakers: int = 2,
    pause_between_files: int = 0
) -> dict:
    """
    Process a list of audio files.

    Args:
        client: AzureBatchTranscription client
        blob_names: List of blob names to process
        output_dir: Directory to save results
        locale: Language locale code (e.g., 'kn-IN')
        enable_diarization: Whether to enable speaker diarization
        enable_transliteration: Whether to enable transliteration
        min_speakers: Minimum speaker count
        max_speakers: Maximum speaker count
        pause_between_files: Seconds to pause between files

    Returns:
        Dictionary with processing results and statistics
    """
    results = []
    failed_files = []
    skipped_files = []

    start_time = datetime.now()

    for idx, blob_name in enumerate(blob_names, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(blob_names)}] {blob_name}")
        print(f"{'='*80}")

        # Check if already processed
        output_file = output_dir / f"{Path(blob_name).stem}_azure_full.json"
        if output_file.exists():
            print(f"⊙ Already processed, skipping")
            skipped_files.append(blob_name)

            # Load existing result for summary
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                    results.append(existing_result)
            except Exception as e:
                print(f"⚠️  Warning: Could not load existing result: {e}")
            continue

        try:
            result = client.transcribe_file(
                blob_name=blob_name,
                locale=locale,
                enable_diarization=enable_diarization,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                enable_transliteration=enable_transliteration,
                transliteration_model="gpt-4o-mini",
                poll_interval=5,
                max_wait_time=600  # 10 minutes max
            )

            results.append(result)

            # Save individual result
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"✓ Saved to: {output_file.name}")

            # Pause if requested
            if pause_between_files > 0 and idx < len(blob_names):
                print(f"⏸️  Pausing for {pause_between_files} seconds...")
                time.sleep(pause_between_files)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Failed: {error_msg}")
            failed_files.append({'file': blob_name, 'error': error_msg})
            continue

    elapsed_time = datetime.now() - start_time

    # Calculate statistics
    new_files_count = len(results) - len(skipped_files)
    total_duration = sum(r.get('duration', 0) for r in results)
    total_azure_cost = sum(r.get('cost', 0) for r in results)
    total_transliteration_cost = sum(r.get('transliteration_cost', 0) for r in results)
    total_cost = total_azure_cost + total_transliteration_cost

    return {
        'new_files_processed': new_files_count,
        'skipped_count': len(skipped_files),
        'failed_count': len(failed_files),
        'total_files': len(results),
        'total_duration_seconds': total_duration,
        'total_duration_hours': total_duration / 3600,
        'azure_cost': total_azure_cost,
        'transliteration_cost': total_transliteration_cost,
        'total_cost': total_cost,
        'elapsed_time_seconds': elapsed_time.total_seconds(),
        'diarization_enabled': enable_diarization,
        'transliteration_enabled': enable_transliteration,
        'failed_files': failed_files,
        'completed_at': datetime.now().isoformat()
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Transcribe audio files from Azure Blob Storage with diarization and transliteration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--file-types',
        type=str,
        help='Comma-separated list of file extensions (e.g., mp3,wav,aac)'
    )

    parser.add_argument(
        '--files',
        type=str,
        help='Comma-separated list of specific filenames to process'
    )

    parser.add_argument(
        '--locale',
        type=str,
        default='kn-IN',
        help='Language locale code (default: kn-IN for Kannada)'
    )

    parser.add_argument(
        '--no-diarization',
        action='store_true',
        help='Disable speaker diarization'
    )

    parser.add_argument(
        '--no-transliteration',
        action='store_true',
        help='Disable Roman transliteration'
    )

    parser.add_argument(
        '--min-speakers',
        type=int,
        default=2,
        help='Minimum speaker count for diarization (default: 2)'
    )

    parser.add_argument(
        '--max-speakers',
        type=int,
        default=2,
        help='Maximum speaker count for diarization (default: 2)'
    )

    parser.add_argument(
        '--pause-between-files',
        type=int,
        default=0,
        help='Seconds to pause between files (default: 0)'
    )

    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Auto-confirm processing without prompting'
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv('.env')

    # Print header
    print("=" * 80)
    print("AZURE BATCH TRANSCRIPTION")
    print("=" * 80)
    print(f"Locale: {args.locale}")
    print(f"Diarization: {'Enabled' if not args.no_diarization else 'Disabled'}")
    print(f"Transliteration: {'Enabled' if not args.no_transliteration else 'Disabled'}")
    if args.pause_between_files:
        print(f"Pause between files: {args.pause_between_files}s")
    print("=" * 80)

    # Initialize clients
    try:
        client = AzureBatchTranscription()
        connection_string = os.getenv('AZURE_STORAGE_CONN_STR')
        container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME')

        if not connection_string or not container_name:
            print("❌ Error: Azure credentials not found in .env")
            print("   Please set AZURE_STORAGE_CONN_STR and AZURE_STORAGE_CONTAINER_NAME")
            sys.exit(1)

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
    except Exception as e:
        print(f"❌ Error initializing Azure clients: {e}")
        sys.exit(1)

    # Parse file types
    file_types = None
    if args.file_types:
        file_types = [ft.strip() for ft in args.file_types.split(',')]
        print(f"\nFile types: {', '.join(file_types)}")

    # Parse specific files
    specific_files = None
    if args.files:
        specific_files = [f.strip() for f in args.files.split(',')]
        print(f"\nSpecific files: {len(specific_files)} files specified")

    # List files to process
    try:
        blob_names = list_audio_files(container_client, file_types, specific_files)
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        sys.exit(1)

    if not blob_names:
        print("\n❌ No files found matching criteria")
        sys.exit(1)

    print(f"\nFound {len(blob_names)} files to process")

    # Show file list
    if len(blob_names) <= 20:
        print("\nFiles to process:")
        for blob_name in blob_names:
            print(f"  - {blob_name}")
    else:
        print("\nFirst 10 files:")
        for blob_name in blob_names[:10]:
            print(f"  - {blob_name}")
        print(f"  ... and {len(blob_names) - 10} more")

    # Confirm if many files
    if len(blob_names) > 10 and not args.files and not args.yes:
        response = input(f"\n⚠️  Process {len(blob_names)} files? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled by user")
            sys.exit(0)

    # Output directory
    output_dir = Path(__file__).parent.parent / "files" / "transcriptions" / "azure_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    print(f"\n{'='*80}")
    print("STARTING BATCH PROCESSING")
    print(f"{'='*80}\n")

    summary = process_files(
        client=client,
        blob_names=blob_names,
        output_dir=output_dir,
        locale=args.locale,
        enable_diarization=not args.no_diarization,
        enable_transliteration=not args.no_transliteration,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        pause_between_files=args.pause_between_files
    )

    # Print summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Elapsed time: {summary['elapsed_time_seconds']/60:.1f} minutes")
    print(f"New files processed: {summary['new_files_processed']}")
    print(f"Already completed: {summary['skipped_count']} files")
    print(f"Failed: {summary['failed_count']} files")

    if summary['total_files'] > 0:
        print(f"\nTotal audio processed (all files): {summary['total_duration_seconds'] / 60:.2f} minutes ({summary['total_duration_hours']:.2f} hours)")
        print(f"Azure Speech cost: ${summary['azure_cost']:.4f}")
        print(f"Transliteration cost: ${summary['transliteration_cost']:.4f}")
        print(f"Total cost: ${summary['total_cost']:.4f}")

    if summary['failed_files']:
        print(f"\n⚠️  Failed files ({len(summary['failed_files'])}):")
        for fail in summary['failed_files']:
            print(f"  - {fail['file']}: {fail['error'][:150]}")

    # Save batch summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = output_dir / f"batch_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_file.name}")
    print(f"\n{'='*80}")
    print("✅ Batch processing complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
