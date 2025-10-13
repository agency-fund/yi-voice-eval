"""
OpenAI GPT-based transliteration for Indic languages.

This module provides functions for transliterating text from native scripts
(e.g., Kannada, Hindi) to Roman/Latin script using OpenAI's GPT models.
"""

import os
import time
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create OpenAI client for API calls.

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


def transliterate_text(
    text: str,
    source_language: str = "Kannada",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0
) -> Tuple[str, Dict[str, Any]]:
    """
    Transliterate text to Roman/Latin script using OpenAI GPT.

    Args:
        text: Text to transliterate (in native script)
        source_language: Name of source language (default: "Kannada")
        model: GPT model to use (default: "gpt-4o-mini")
        api_key: OpenAI API key (optional, reads from env if not provided)
        system_prompt: Custom system prompt (optional)
        temperature: Model temperature (default: 0.0 for deterministic output)

    Returns:
        Tuple of (transliterated_text, metrics_dict)

        metrics_dict contains:
        - tokens: dict with input, output, and total token counts
        - latency_ms: Response time in milliseconds
        - cost_usd: Estimated cost in USD

    Raises:
        ValueError: If text is empty or API key not available

    Example:
        >>> romanized, metrics = transliterate_text("ಕನ್ನಡ")
        >>> print(romanized)
        kannada
        >>> print(f"Cost: ${metrics['cost_usd']:.6f}")
        Cost: $0.000029
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    # Get client
    client = get_openai_client(api_key)

    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = (
            f"You are a {source_language} language expert specializing in transliteration. "
            f"Transliterate the provided {source_language} text to Roman/Latin script. "
            "Use standard phonetic romanization that is readable and pronounceable. "
            "Preserve the original meaning and pronunciation as closely as possible. "
            "Only output the transliterated text, no explanations."
        )

    # Time the API call
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=temperature
    )

    latency_ms = (time.time() - start_time) * 1000

    # Extract token usage
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    # Calculate cost based on model pricing
    cost_usd = _calculate_cost(model, input_tokens, output_tokens)

    metrics = {
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens
        },
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "model": model
    }

    transliterated_text = response.choices[0].message.content.strip()

    return transliterated_text, metrics


def transliterate_segments(
    segments: List[Dict[str, Any]],
    text_field: str = "text",
    source_language: str = "Kannada",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Transliterate text in a list of segments (e.g., from STT output).

    Adds 'text_romanized' field to each segment with the transliterated text.

    Args:
        segments: List of segment dictionaries with text to transliterate
        text_field: Name of field containing text to transliterate (default: "text")
        source_language: Name of source language (default: "Kannada")
        model: GPT model to use (default: "gpt-4o-mini")
        api_key: OpenAI API key (optional)
        verbose: Print progress updates every 10 segments (default: True)

    Returns:
        Tuple of (updated_segments, aggregate_metrics)

        updated_segments: Original segments with added 'text_romanized' field
        aggregate_metrics: Dictionary with:
            - total_segments: Number of segments processed
            - total_cost_usd: Total cost for all segments
            - total_tokens: Total tokens used
            - total_latency_ms: Total processing time
            - avg_latency_per_segment_ms: Average time per segment

    Example:
        >>> segments = [
        ...     {"start": 0.0, "end": 2.0, "text": "ಕನ್ನಡ"},
        ...     {"start": 2.0, "end": 4.0, "text": "ಭಾಷೆ"}
        ... ]
        >>> updated, metrics = transliterate_segments(segments)
        >>> print(updated[0]['text_romanized'])
        kannada
        >>> print(f"Total cost: ${metrics['total_cost_usd']:.6f}")
    """
    updated_segments = []
    total_cost = 0.0
    total_tokens = 0
    total_latency = 0.0

    for i, segment in enumerate(segments, 1):
        text = segment.get(text_field, "")

        if text.strip():
            try:
                romanized, metrics = transliterate_text(
                    text=text,
                    source_language=source_language,
                    model=model,
                    api_key=api_key
                )
                segment["text_romanized"] = romanized

                total_cost += metrics["cost_usd"]
                total_tokens += metrics["tokens"]["total"]
                total_latency += metrics["latency_ms"]

                # Progress indicator
                if verbose and i % 10 == 0:
                    print(f"  Processed {i}/{len(segments)} segments...")

            except Exception as e:
                # Log error but continue processing
                print(f"  Warning: Failed to transliterate segment {i}: {e}")
                segment["text_romanized"] = f"[Error: {str(e)}]"
        else:
            segment["text_romanized"] = ""

        updated_segments.append(segment)

    aggregate_metrics = {
        "total_segments": len(segments),
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency,
        "avg_latency_per_segment_ms": total_latency / len(segments) if segments else 0
    }

    return updated_segments, aggregate_metrics


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate estimated cost based on model pricing.

    Pricing as of January 2025:
    - gpt-4o-mini: $0.15/1M input, $0.60/1M output
    - gpt-4o: $2.50/1M input, $10.00/1M output

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    # Pricing per 1M tokens (USD)
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }

    # Default to gpt-4o-mini pricing if model not recognized
    model_pricing = pricing.get(model, pricing["gpt-4o-mini"])

    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

    return input_cost + output_cost


def estimate_transliteration_cost(
    num_characters: int,
    model: str = "gpt-4o-mini",
    chars_per_token: float = 3.0
) -> float:
    """
    Estimate transliteration cost for a given amount of text.

    Args:
        num_characters: Number of characters to transliterate
        model: GPT model to use (default: "gpt-4o-mini")
        chars_per_token: Average characters per token (default: 3.0 for Indic scripts)

    Returns:
        Estimated cost in USD

    Example:
        >>> # Estimate cost for 10,000 characters
        >>> cost = estimate_transliteration_cost(10_000)
        >>> print(f"Estimated: ${cost:.4f}")
    """
    # Rough estimates
    input_tokens = int(num_characters / chars_per_token) + 100  # +100 for system prompt
    output_tokens = int(num_characters / chars_per_token)  # Similar length in Roman script

    return _calculate_cost(model, input_tokens, output_tokens)
