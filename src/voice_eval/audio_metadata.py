"""Audio metadata extraction using ffprobe."""

import subprocess
import json
from pathlib import Path


def get_audio_metadata(file_path: str) -> dict:
    """Extract metadata using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)


def extract_audio_info(file_path: str) -> dict:
    """Extract common audio file information."""
    info = get_audio_metadata(file_path)
    format_info = info.get('format', {})

    audio_stream = next(
        (s for s in info.get('streams', []) if s['codec_type'] == 'audio'),
        {}
    )

    return {
        'file': Path(file_path).name,
        'duration': float(format_info.get('duration', 0)),
        'size_mb': float(format_info.get('size', 0)) / 1024 / 1024,
        'format': format_info.get('format_name', 'unknown'),
        'codec': audio_stream.get('codec_name', 'unknown'),
        'sample_rate': audio_stream.get('sample_rate', 'unknown'),
        'channels': audio_stream.get('channels', 'unknown'),
        'bit_rate': int(format_info.get('bit_rate', 0)) / 1000,
    }
