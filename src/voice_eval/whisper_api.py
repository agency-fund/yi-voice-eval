"""
OpenAI Whisper API integration for speech-to-text transcription.

This module provides functions for transcribing audio files using the OpenAI
Whisper API (whisper-1 model) instead of running Whisper locally.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from openai import OpenAI
from openai.types.audio import Transcription


def get_whisper_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create OpenAI client for Whisper API calls.

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.

    Returns:
        Configured OpenAI client

    Raises:
        ValueError: If API key not provided and not in environment
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Either pass api_key parameter or "
            "set OPENAI_API_KEY environment variable."
        )

    return OpenAI(api_key=api_key)


def transcribe_audio(
    audio_path: str,
    language: str = "kn",
    model: str = "whisper-1",
    api_key: Optional[str] = None,
    response_format: str = "verbose_json",
    timestamp_granularities: Optional[list] = None
) -> Transcription:
    """
    Transcribe audio file using OpenAI Whisper API.

    Args:
        audio_path: Path to audio file (supports mp3, mp4, mpeg, mpga, m4a, wav, webm)
        language: ISO-639-1 language code (default: "kn" for Kannada)
        model: Whisper model to use (default: "whisper-1")
        api_key: OpenAI API key (optional, reads from env if not provided)
        response_format: Format of response - "json", "text", "srt", "verbose_json", "vtt"
        timestamp_granularities: List of timestamp granularities (e.g., ["segment"])

    Returns:
        Transcription object with text, language, duration, and segments (if verbose_json)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If file format not supported

    Example:
        >>> result = transcribe_audio("audio.mp3", language="kn")
        >>> print(result.text)  # Full transcription
        >>> for seg in result.segments:
        ...     print(f"{seg.start:.2f}s: {seg.text}")
    """
    # Validate file exists
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get client
    client = get_whisper_client(api_key)

    # Prepare API call parameters
    params: Dict[str, Any] = {
        "model": model,
        "language": language,
        "response_format": response_format,
    }

    # Add timestamp granularities if provided and format supports it
    if timestamp_granularities and response_format == "verbose_json":
        params["timestamp_granularities"] = timestamp_granularities

    # Transcribe
    with open(audio_path, 'rb') as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            **params
        )

    return transcription


def transcription_to_dict(transcription: Transcription) -> dict:
    """
    Convert Transcription object to a serializable dictionary.

    Useful for saving to JSON files.

    Args:
        transcription: Transcription object from Whisper API

    Returns:
        Dictionary with text, language, duration, and segments
    """
    result = {
        "text": transcription.text,
        "language": transcription.language,
        "duration": transcription.duration,
    }

    # Add segments if available (verbose_json format)
    if hasattr(transcription, 'segments') and transcription.segments:
        result["segments"] = [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "tokens": seg.tokens,
                "temperature": seg.temperature,
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob,
            }
            for seg in transcription.segments
        ]

    return result


def estimate_cost(duration_seconds: float, cost_per_minute: float = 0.006) -> float:
    """
    Estimate transcription cost for given audio duration.

    OpenAI Whisper API pricing (as of 2025): $0.006 per minute

    Args:
        duration_seconds: Audio duration in seconds
        cost_per_minute: Cost per minute (default: $0.006)

    Returns:
        Estimated cost in USD

    Example:
        >>> estimate_cost(600)  # 10 minutes
        0.06
    """
    duration_minutes = duration_seconds / 60.0
    return duration_minutes * cost_per_minute
