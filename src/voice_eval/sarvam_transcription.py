"""Sarvam AI transcription utilities."""

import os
import requests
from pathlib import Path
from typing import Optional


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
