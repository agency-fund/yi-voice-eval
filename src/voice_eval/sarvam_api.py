"""
Sarvam AI API integration for speech-to-text and transliteration.

Sarvam AI is an India-focused AI platform that provides:
- Speech-to-text for Indian languages (including Kannada)
- Transliteration from native scripts to Roman/Latin script

This module provides functions for both STT and transliteration.
"""

import os
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any


def transcribe_with_sarvam(
    audio_path: str,
    api_key: str,
    language_code: str = "kn-IN",
    transliterate: bool = True
) -> dict:
    """
    Transcribe audio file using Sarvam AI API.

    Args:
        audio_path: Path to audio file
        api_key: Sarvam API key
        language_code: Language code (default: kn-IN for Kannada)
        transliterate: Whether to also transliterate to Roman script

    Returns:
        dict with 'kannada', 'romanized', 'request_id', 'language_code' keys
    """
    url = "https://api.sarvam.ai/speech-to-text"

    headers = {
        "api-subscription-key": api_key
    }

    with open(audio_path, 'rb') as f:
        files = {
            'file': (os.path.basename(audio_path), f, 'audio/mp4')
        }
        data = {
            'language_code': language_code
        }

        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"Sarvam STT failed: {response.status_code} - {response.text}")

    result = response.json()
    kannada_transcript = result.get('transcript', '')

    output = {
        'kannada': kannada_transcript,
        'romanized': None,
        'request_id': result.get('request_id'),
        'language_code': result.get('language_code')
    }

    if transliterate and kannada_transcript:
        romanized = transliterate_text(kannada_transcript, api_key, language_code)
        output['romanized'] = romanized

    return output


def transliterate_text(
    text: str,
    api_key: str,
    source_language_code: str = "kn-IN"
) -> Optional[str]:
    """
    Transliterate text to Roman script using Sarvam API.

    Args:
        text: Text to transliterate
        api_key: Sarvam API key
        source_language_code: Source language code

    Returns:
        Transliterated text or None if failed
    """
    url = "https://api.sarvam.ai/transliterate"

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "input": text,
        "source_language_code": source_language_code,
        "target_language_code": "en-IN"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return None

    result = response.json()
    return result.get('transliterated_text')


def transliterate_segments(
    segments: List[Dict[str, Any]],
    api_key: str,
    source_language_code: str = "kn-IN",
    text_field: str = "text"
) -> List[Dict[str, Any]]:
    """
    Transliterate text in a list of segments (e.g., from Whisper API).

    Useful for transliterating Whisper's timestamped segments.

    Args:
        segments: List of segment dicts containing text to transliterate
        api_key: Sarvam API key
        source_language_code: Source language code
        text_field: Key in segment dict containing text to transliterate

    Returns:
        List of segments with added 'text_romanized' field

    Example:
        >>> segments = [{"text": "ಕನ್ನಡ", "start": 0.0, "end": 1.0}]
        >>> result = transliterate_segments(segments, api_key)
        >>> print(result[0]["text_romanized"])
        "Kannada"
    """
    romanized_segments = []

    for segment in segments:
        # Copy original segment
        new_segment = segment.copy()

        # Transliterate text if present
        if text_field in segment and segment[text_field]:
            romanized = transliterate_text(
                segment[text_field],
                api_key,
                source_language_code
            )
            new_segment["text_romanized"] = romanized or segment[text_field]
        else:
            new_segment["text_romanized"] = None

        romanized_segments.append(new_segment)

    return romanized_segments


def get_sarvam_api_key() -> str:
    """
    Get Sarvam API key from environment variable.

    Returns:
        API key string

    Raises:
        ValueError: If SARVAM_API_KEY not found in environment
    """
    api_key = os.getenv('SARVAM_API_KEY')
    if not api_key:
        raise ValueError(
            "SARVAM_API_KEY not found in environment. "
            "Set it in your .env file."
        )
    return api_key


def save_transcription_as_text(
    transcription: dict,
    output_path: str,
    include_romanized: bool = True
) -> None:
    """
    Save Sarvam transcription to text file.

    Args:
        transcription: dict from transcribe_with_sarvam()
        output_path: Path to save file
        include_romanized: Whether to include romanized version
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Sarvam AI Transcription\n\n")
        f.write(f"Request ID: {transcription.get('request_id')}\n")
        f.write(f"Language: {transcription.get('language_code')}\n\n")

        f.write("## Kannada (Native Script)\n\n")
        f.write(transcription.get('kannada', '') + "\n\n")

        if include_romanized and transcription.get('romanized'):
            f.write("## Romanized (Latin Script)\n\n")
            f.write(transcription.get('romanized', '') + "\n")
