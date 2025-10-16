"""
Azure Speech Service Batch Transcription API integration.

This module provides functions for transcribing audio files using Azure's
Batch Transcription REST API v3.1, which supports features like diarization
(speaker separation) for Kannada speech-to-text.

Documentation:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription
"""

import os
import time
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json
from urllib.parse import quote


class AzureBatchTranscription:
    """
    Wrapper for Azure Speech Service Batch Transcription REST API v3.1.

    This class handles the full lifecycle of batch transcription jobs:
    1. Creating transcription jobs from audio URLs
    2. Polling job status until completion
    3. Downloading and parsing results
    4. Converting to standardized format matching Whisper output
    """

    def __init__(
        self,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        storage_account: Optional[str] = None,
        container_name: Optional[str] = None,
        sas_token: Optional[str] = None
    ):
        """
        Initialize Azure Batch Transcription client.

        Args:
            speech_key: Azure Speech Service API key (or from AZURE_AI_KEY env)
            speech_region: Azure region (or from AZURE_REGION env)
            storage_account: Azure Storage account name (or from AZURE_STORAGE_BUCKET_NAME env)
            container_name: Blob container name (or from AZURE_STORAGE_CONTAINER_NAME env)
            sas_token: SAS token for blob access (or from AZURE_STORAGE_SAS_TOKEN env)

        Raises:
            ValueError: If required credentials are missing
        """
        # Get credentials from params or environment
        self.speech_key = speech_key or os.getenv('AZURE_AI_KEY')
        self.speech_region = speech_region or os.getenv('AZURE_REGION')
        self.storage_account = storage_account or os.getenv('AZURE_STORAGE_BUCKET_NAME')
        self.container_name = container_name or os.getenv('AZURE_STORAGE_CONTAINER_NAME')
        self.sas_token = sas_token or os.getenv('AZURE_STORAGE_SAS_TOKEN')

        # Validate required credentials
        if not self.speech_key:
            raise ValueError("Azure Speech key not found. Set AZURE_AI_KEY environment variable.")
        if not self.speech_region:
            raise ValueError("Azure region not found. Set AZURE_REGION environment variable.")
        if not self.storage_account:
            raise ValueError("Storage account not found. Set AZURE_STORAGE_BUCKET_NAME environment variable.")
        if not self.container_name:
            raise ValueError("Container name not found. Set AZURE_STORAGE_CONTAINER_NAME environment variable.")
        if not self.sas_token:
            raise ValueError("SAS token not found. Set AZURE_STORAGE_SAS_TOKEN environment variable.")

        # Ensure SAS token starts with ?
        if not self.sas_token.startswith('?'):
            self.sas_token = '?' + self.sas_token

        # Construct API base URL
        self.base_url = f"https://{self.speech_region}.api.cognitive.microsoft.com/speechtotext/v3.1"

        # Standard headers for API requests
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.speech_key,
            "Content-Type": "application/json"
        }

    def construct_blob_url(self, blob_name: str) -> str:
        """
        Construct full blob URL with SAS token for a given file.

        Args:
            blob_name: Name of the blob (filename) in the container

        Returns:
            Full HTTPS URL with SAS token for blob access
        """
        # URL-encode the blob name to handle special characters (phone numbers, etc.)
        encoded_blob_name = quote(blob_name, safe='')
        return f"https://{self.storage_account}.blob.core.windows.net/{self.container_name}/{encoded_blob_name}{self.sas_token}"

    def create_transcription(
        self,
        audio_url: str,
        locale: str = "kn-IN",
        display_name: Optional[str] = None,
        enable_diarization: bool = False,
        min_speakers: int = 2,
        max_speakers: int = 2,
        enable_word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Create a batch transcription job.

        Args:
            audio_url: Full URL to audio file (must be publicly accessible or have SAS token)
            locale: Language locale code (e.g., "kn-IN" for Kannada)
            display_name: Human-readable name for the job (defaults to filename)
            enable_diarization: Whether to enable speaker diarization (separation)
            min_speakers: Minimum number of speakers (used if diarization enabled)
            max_speakers: Maximum number of speakers (used if diarization enabled)
            enable_word_timestamps: Whether to include word-level timestamps

        Returns:
            API response with job details including self URL for polling

        Raises:
            requests.HTTPError: If API request fails
        """
        # Generate display name from URL if not provided
        if not display_name:
            display_name = Path(audio_url.split('?')[0]).name  # Remove SAS params

        # Construct request body
        request_body = {
            "contentUrls": [audio_url],
            "locale": locale,
            "displayName": display_name,
            "properties": {
                "wordLevelTimestampsEnabled": enable_word_timestamps,
                "punctuationMode": "DictatedAndAutomatic",
                "profanityFilterMode": "None"  # Don't filter any content
            }
        }

        # Add diarization settings if enabled
        if enable_diarization:
            request_body["properties"]["diarizationEnabled"] = True
            request_body["properties"]["diarization"] = {
                "speakers": {
                    "minCount": min_speakers,
                    "maxCount": max_speakers
                }
            }

        # Make POST request to create transcription
        response = requests.post(
            f"{self.base_url}/transcriptions",
            headers=self.headers,
            json=request_body
        )
        response.raise_for_status()

        return response.json()

    def get_transcription_status(self, transcription_url: str) -> Dict[str, Any]:
        """
        Get current status of a transcription job.

        Args:
            transcription_url: The 'self' URL returned from create_transcription

        Returns:
            Status object with 'status' field: 'NotStarted', 'Running', 'Succeeded', 'Failed'
        """
        response = requests.get(transcription_url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def wait_for_completion(
        self,
        transcription_url: str,
        poll_interval: int = 5,
        max_wait_time: int = 600
    ) -> Dict[str, Any]:
        """
        Poll transcription job status until it completes (or fails).

        Args:
            transcription_url: The 'self' URL from create_transcription
            poll_interval: Seconds to wait between status checks
            max_wait_time: Maximum seconds to wait before timing out

        Returns:
            Final transcription status object

        Raises:
            TimeoutError: If job doesn't complete within max_wait_time
            RuntimeError: If job fails
        """
        start_time = time.time()

        while True:
            status_obj = self.get_transcription_status(transcription_url)
            status = status_obj.get('status')

            print(f"  Status: {status}", end='\r')

            if status == 'Succeeded':
                print(f"  Status: {status} ✓")
                return status_obj
            elif status == 'Failed':
                error_msg = status_obj.get('properties', {}).get('error', {})
                raise RuntimeError(f"Transcription failed: {error_msg}")
            elif time.time() - start_time > max_wait_time:
                raise TimeoutError(f"Transcription timed out after {max_wait_time}s")

            time.sleep(poll_interval)

    def get_transcription_files(self, transcription_url: str) -> List[Dict[str, Any]]:
        """
        Get list of result files for a completed transcription.

        Args:
            transcription_url: The 'self' URL from create_transcription

        Returns:
            List of file objects, each with 'kind' and 'links' (contentUrl)
        """
        files_url = transcription_url + "/files"
        response = requests.get(files_url, headers=self.headers)
        response.raise_for_status()
        return response.json().get('values', [])

    def download_transcription_result(self, transcription_url: str) -> Dict[str, Any]:
        """
        Download and parse the transcription result JSON.

        Args:
            transcription_url: The 'self' URL from create_transcription

        Returns:
            Parsed transcription result with segments and speaker info
        """
        files = self.get_transcription_files(transcription_url)

        # Find the transcription result file (kind: 'Transcription')
        result_file = next(
            (f for f in files if f.get('kind') == 'Transcription'),
            None
        )

        if not result_file:
            raise ValueError("No transcription result file found")

        # Download the result JSON
        content_url = result_file['links']['contentUrl']
        response = requests.get(content_url)
        response.raise_for_status()

        return response.json()

    def parse_azure_result(
        self,
        azure_result: Dict[str, Any],
        filename: str
    ) -> Dict[str, Any]:
        """
        Convert Azure transcription result to standardized format matching Whisper output.

        Args:
            azure_result: Raw Azure API response JSON
            filename: Original audio filename

        Returns:
            Dictionary matching Whisper API format with segments, timestamps, etc.
        """
        # Extract recognized phrases
        phrases = azure_result.get('recognizedPhrases', [])

        # Combine all text
        full_text = ' '.join(
            phrase.get('nBest', [{}])[0].get('display', '')
            for phrase in phrases
        )

        # Extract duration (in ticks: 1 tick = 100 nanoseconds)
        duration_ticks = azure_result.get('duration', 0)
        duration_seconds = self._parse_duration(duration_ticks)

        # Convert phrases to segments matching Whisper format
        segments = []
        for idx, phrase in enumerate(phrases):
            best_result = phrase.get('nBest', [{}])[0]

            segment = {
                'id': idx,
                'start': self._parse_duration(phrase.get('offset', 0)),
                'end': self._parse_duration(phrase.get('offset', 0)) + self._parse_duration(phrase.get('duration', 0)),
                'text': best_result.get('display', ''),
                'confidence': best_result.get('confidence', 0.0)
            }

            # Add speaker ID if diarization was enabled
            if 'speaker' in phrase:
                segment['speaker'] = phrase['speaker']

            segments.append(segment)

        # Calculate cost (Azure Speech: ~$1 per audio hour for Standard tier)
        cost = (duration_seconds / 3600) * 1.00  # $1/hour

        return {
            'file': filename,
            'duration': duration_seconds,
            'language': azure_result.get('locale', 'kn-IN').split('-')[0],  # 'kn-IN' -> 'kn'
            'text': full_text.strip(),
            'segments': segments,
            'model': f"azure-speech-{azure_result.get('locale', 'kn-IN')}",
            'cost': round(cost, 4),
            'processed_at': datetime.now().isoformat()
        }

    def _parse_duration(self, duration_str: str) -> float:
        """
        Parse Azure duration format (ISO 8601 or ticks) to seconds.

        Azure uses formats like:
        - "PT5.2S" (5.2 seconds)
        - "PT7M58.71S" (7 minutes 58.71 seconds)
        - "PT1H30M15.5S" (1 hour 30 minutes 15.5 seconds)
        - Integer ticks (1 tick = 100 nanoseconds)

        Args:
            duration_str: Duration string or integer

        Returns:
            Duration in seconds (float)
        """
        if isinstance(duration_str, (int, float)):
            # Assume ticks (100 nanoseconds each)
            return duration_str / 10_000_000  # Convert to seconds

        if isinstance(duration_str, str):
            # Parse ISO 8601 duration format: PT[H]H[M]M[S]S
            if duration_str.startswith('PT') and duration_str.endswith('S'):
                import re

                # Remove PT prefix and S suffix
                duration_content = duration_str[2:-1]

                # Extract hours, minutes, seconds using regex
                hours = 0.0
                minutes = 0.0
                seconds = 0.0

                # Match hours: digits followed by H
                hours_match = re.search(r'(\d+(?:\.\d+)?)H', duration_content)
                if hours_match:
                    hours = float(hours_match.group(1))

                # Match minutes: digits followed by M
                minutes_match = re.search(r'(\d+(?:\.\d+)?)M', duration_content)
                if minutes_match:
                    minutes = float(minutes_match.group(1))

                # Match seconds: remaining digits (no letter or just before end)
                # Remove H and M parts first
                seconds_part = duration_content
                seconds_part = re.sub(r'\d+(?:\.\d+)?H', '', seconds_part)
                seconds_part = re.sub(r'\d+(?:\.\d+)?M', '', seconds_part)
                if seconds_part:
                    seconds = float(seconds_part)

                return hours * 3600 + minutes * 60 + seconds

        return 0.0

    def transcribe_file(
        self,
        blob_name: str,
        locale: str = "kn-IN",
        enable_diarization: bool = False,
        min_speakers: int = 2,
        max_speakers: int = 2,
        enable_transliteration: bool = False,
        transliteration_model: str = "gpt-4o-mini",
        poll_interval: int = 5,
        max_wait_time: int = 600
    ) -> Dict[str, Any]:
        """
        High-level function: Transcribe a single audio file (end-to-end).

        This function handles the complete workflow:
        1. Construct blob URL from filename
        2. Create transcription job
        3. Poll until completion
        4. Download and parse results
        5. Convert to standardized format
        6. Optionally transliterate to Roman script (using GPT)

        Args:
            blob_name: Name of audio file in blob storage (e.g., "file.mp3")
            locale: Language locale (default: "kn-IN" for Kannada)
            enable_diarization: Enable speaker separation
            min_speakers: Minimum speakers (for diarization)
            max_speakers: Maximum speakers (for diarization)
            enable_transliteration: Transliterate Kannada to Roman script (default: False)
            transliteration_model: GPT model for transliteration (default: "gpt-4o-mini")
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait for completion

        Returns:
            Standardized transcription result matching Whisper format
            (with optional 'romanized_text' and 'text_romanized' fields)
        """
        print(f"Transcribing: {blob_name}")

        # Step 1: Construct blob URL
        audio_url = self.construct_blob_url(blob_name)

        # Step 2: Create transcription job
        print("  Creating transcription job...")
        job_response = self.create_transcription(
            audio_url=audio_url,
            locale=locale,
            display_name=blob_name,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        transcription_url = job_response['self']
        # Extract ID from URL (last part after final /)
        job_id = transcription_url.split('/')[-1]
        print(f"  Job created: {job_id}")

        # Step 3: Wait for completion
        print("  Waiting for transcription to complete...")
        final_status = self.wait_for_completion(
            transcription_url,
            poll_interval=poll_interval,
            max_wait_time=max_wait_time
        )

        # Step 4: Download results
        print("  Downloading results...")
        azure_result = self.download_transcription_result(transcription_url)

        # Step 5: Convert to standardized format
        print("  Parsing results...")
        standardized_result = self.parse_azure_result(azure_result, blob_name)

        # Step 6: Optional transliteration to Roman script
        if enable_transliteration:
            print("  Transliterating to Roman script...")
            try:
                from voice_eval.gpt_transliteration import transliterate_text, transliterate_segments

                # Transliterate full text
                romanized_text, full_text_metrics = transliterate_text(
                    text=standardized_result['text'],
                    source_language="Kannada",
                    model=transliteration_model
                )
                standardized_result['romanized_text'] = romanized_text

                # Transliterate segments
                updated_segments, segment_metrics = transliterate_segments(
                    segments=standardized_result['segments'],
                    text_field='text',
                    source_language="Kannada",
                    model=transliteration_model,
                    verbose=False  # Suppress progress messages
                )
                standardized_result['segments'] = updated_segments

                # Add transliteration costs to result
                total_transliteration_cost = full_text_metrics['cost_usd'] + segment_metrics['total_cost_usd']
                standardized_result['transliteration_cost'] = round(total_transliteration_cost, 6)
                standardized_result['transliteration_model'] = transliteration_model

                print(f"  ✓ Transliteration complete: ${total_transliteration_cost:.6f}")

            except Exception as e:
                print(f"  ⚠️  Transliteration failed: {e}")
                standardized_result['romanized_text'] = f"[Transliteration error: {str(e)}]"

        print(f"  ✓ Transcription complete: {standardized_result['duration']:.1f}s, ${standardized_result['cost']:.4f}")

        return standardized_result


def transcribe_audio(
    blob_name: str,
    locale: str = "kn-IN",
    enable_diarization: bool = False,
    min_speakers: int = 2,
    max_speakers: int = 2,
    enable_transliteration: bool = False,
    transliteration_model: str = "gpt-4o-mini",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function: Transcribe audio file using Azure Batch API.

    Args:
        blob_name: Audio filename in Azure blob storage
        locale: Language locale code (default: "kn-IN")
        enable_diarization: Enable speaker diarization
        min_speakers: Minimum speaker count
        max_speakers: Maximum speaker count
        enable_transliteration: Transliterate to Roman script (default: False)
        transliteration_model: GPT model for transliteration (default: "gpt-4o-mini")
        **kwargs: Additional arguments passed to AzureBatchTranscription

    Returns:
        Transcription result in standardized format
        (with optional 'romanized_text' and 'text_romanized' fields)

    Example:
        >>> result = transcribe_audio("audio.mp3",
        ...                          enable_diarization=True,
        ...                          enable_transliteration=True)
        >>> print(result['text'])  # Kannada script
        >>> print(result['romanized_text'])  # Roman script
    """
    client = AzureBatchTranscription()
    return client.transcribe_file(
        blob_name=blob_name,
        locale=locale,
        enable_diarization=enable_diarization,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        enable_transliteration=enable_transliteration,
        transliteration_model=transliteration_model,
        **kwargs
    )
