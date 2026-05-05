#!/usr/bin/env python3
"""
VibeVoice OpenAI-Compatible TTS Server

A FastAPI server that wraps VibeVoice-Realtime-0.5B with an OpenAI-compatible API,
enabling integration with Open WebUI and other OpenAI TTS-compatible applications.

Usage:
    python vibevoice_realtime_openai_api.py --port 8880
"""

import argparse
import copy
import io
import os
import shutil
import subprocess
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from contextlib import asynccontextmanager, nullcontext, redirect_stderr

# Set HuggingFace cache BEFORE importing any HF libraries
# Only use HF_HOME (TRANSFORMERS_CACHE is deprecated in v5)
# MODELS_DIR can be overridden via env var for Docker volume mounts
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import scipy.io.wavfile as wavfile

# VibeVoice imports (after setting HF_HOME)
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"

# CFG scale for generation (configurable via env var)
CFG_SCALE = float(os.environ.get("CFG_SCALE", "1.25"))

# Output volume gain. 1.0 is unchanged, 0.5 is half, 2.0 is louder.
DEFAULT_VOLUME = float(os.environ.get(
    "VIBEVOICE_VOLUME",
    os.environ.get(
        "DEFAULT_VOLUME_MULTIPLIER",
        os.environ.get(
            "default_volume_multiplier",
            os.environ.get("volume_multiplier", os.environ.get("VOLUME", "1.0"))
        )
    )
))

# Voices directory
VOICES_DIR = MODELS_DIR / "voices"

# Voice presets available for download
VOICE_PRESETS = {
    "Carter": "en-Carter_man.pt",
    "Davis": "en-Davis_man.pt",
    "Emma": "en-Emma_woman.pt",
    "Frank": "en-Frank_man.pt",
    "Grace": "en-Grace_woman.pt",
    "Mike": "en-Mike_man.pt",
    "Samuel": "in-Samuel_man.pt",
}

# GitHub raw URL for voice presets
VOICE_BASE_URL = "https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model"

# OpenAI voice name mapping to VibeVoice voices
OPENAI_TO_VIBEVOICE_MAP = {
    "alloy": "Carter",
    "ash": "Davis",
    "ballad": "Emma",
    "coral": "Grace",
    "echo": "Davis",
    "fable": "Emma",
    "marin": "Grace",
    "onyx": "Frank",
    "nova": "Grace",
    "sage": "Carter",
    "shimmer": "Mike",
    "verse": "Frank",
}

# Supported audio formats
SUPPORTED_FORMATS = ["mp3", "wav", "opus", "flac", "aac", "pcm"]
STREAMABLE_FORMATS = ["mp3", "opus", "flac", "aac", "pcm"]

# ffmpeg format mappings
FFMPEG_FORMAT_ARGS = {
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"],
    "opus": ["-f", "opus", "-codec:a", "libopus"],
    "flac": ["-f", "flac", "-codec:a", "flac"],
    "aac": ["-f", "adts", "-codec:a", "aac"],
}

# ------------------------------------------------------------------------------
# Model Download Utilities
# ------------------------------------------------------------------------------

def ensure_voices_downloaded() -> None:
    """Download voice presets if not present"""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    for voice_name, filename in VOICE_PRESETS.items():
        voice_path = VOICES_DIR / filename
        if not voice_path.exists():
            url = f"{VOICE_BASE_URL}/{filename}"
            print(f"[download] Downloading voice preset: {voice_name}...")
            try:
                urllib.request.urlretrieve(url, voice_path)
                print(f"[download] Downloaded {filename}")
            except Exception as e:
                print(f"[error] Failed to download {filename}: {e}")


def get_model_cache_dir() -> str:
    """Get model cache directory"""
    model_cache = MODELS_DIR / "huggingface"
    model_cache.mkdir(parents=True, exist_ok=True)
    return str(model_cache)


# ------------------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------------------

class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request"""
    input: str = Field(..., description="Text to synthesize", max_length=4096)
    voice: str = Field(default="Carter", description="Voice ID")
    model: str = Field(default="tts-1", description="Model ID (ignored, for compatibility)")
    instructions: Optional[str] = Field(default=None, description="Voice instructions (ignored)")
    response_format: str = Field(default="mp3", description="Audio format")
    speed: float = Field(default=1.0, description="Speed (not yet supported)")
    volume: Optional[float] = Field(default=None, ge=0.0, le=4.0, description="Output volume gain")
    volume_multiplier: Optional[float] = Field(default=None, ge=0.0, le=4.0, description="Output volume gain")
    stream: bool = Field(default=False, description="Enable streaming response")


class VoiceInfo(BaseModel):
    """Voice information"""
    voice_id: str
    name: str
    type: str
    gender: Optional[str] = None


class VoicesResponse(BaseModel):
    """Response for /v1/audio/voices endpoint"""
    voices: List[VoiceInfo]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    model_loaded: bool
    device: str
    features: Dict[str, Any]


# ------------------------------------------------------------------------------
# TTS Service
# ------------------------------------------------------------------------------

class VibeVoiceTTSService:
    """Service for managing VibeVoice model and generating speech"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self._voice_cache: Dict[str, Any] = {}
        self.device = self._resolve_device(device)
        self._torch_device = torch.device(self.device)

    def _resolve_device(self, device: str) -> str:
        """Resolve auto/cuda/mps/cpu to an available torch device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            print("[startup] CUDA requested but not available; falling back to CPU")
            return "cpu"

        if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("[startup] MPS requested but not available; falling back to CPU")
            return "cpu"

        return device

    def load(self) -> None:
        """Load model and voice presets"""
        # Set HuggingFace cache to models folder
        os.environ["HF_HOME"] = get_model_cache_dir()

        # Download voice presets
        ensure_voices_downloaded()

        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # Determine dtype and attention implementation based on device
        if self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl = "flash_attention_2"
        elif self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl = "sdpa"
        else:  # cpu
            load_dtype = torch.float32
            device_map = "cpu"
            attn_impl = "sdpa"

        print(f"[startup] Loading model with dtype={load_dtype}, attn={attn_impl}")

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"[startup] Flash Attention failed, falling back to SDPA: {e}")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self.model.eval()
        self._configure_noise_scheduler()
        self.model.set_ddpm_inference_steps(num_steps=5)

        # Load voice presets
        self._load_voice_presets()
        if "Carter" in self.voice_presets:
            self._get_voice_prompt("Carter")
        print(f"[startup] Model ready on {self.device}")

    def _configure_noise_scheduler(self) -> None:
        """Use the scheduler settings from the upstream realtime web demo when available."""
        if not self.model or not hasattr(self.model, "model"):
            return

        model_core = self.model.model
        scheduler = getattr(model_core, "noise_scheduler", None)
        if scheduler is None or not hasattr(scheduler, "from_config"):
            return

        try:
            model_core.noise_scheduler = scheduler.from_config(
                scheduler.config,
                algorithm_type="sde-dpmsolver++",
                beta_schedule="squaredcos_cap_v2",
            )
        except Exception as e:
            print(f"[startup] Could not apply realtime scheduler config: {e}")

    def _load_voice_presets(self) -> None:
        """Scan and load available voice presets"""
        if not VOICES_DIR.exists():
            print(f"[warning] Voices directory not found: {VOICES_DIR}")
            return

        for pt_file in VOICES_DIR.glob("*.pt"):
            full_name = pt_file.stem  # e.g., "en-Carter_man"
            self.voice_presets[full_name] = pt_file

            # Also add short name (e.g., "Carter")
            short_name = full_name
            if "_" in short_name:
                short_name = short_name.split("_")[0]
            if "-" in short_name:
                short_name = short_name.split("-")[-1]
            self.voice_presets[short_name] = pt_file

        print(f"[startup] Found {len(self.voice_presets)} voice presets")

    def get_available_voices(self) -> List[VoiceInfo]:
        """Get list of available voices"""
        voices = []
        seen = set()

        # Add OpenAI-compatible voices
        for openai_name, vibevoice_name in OPENAI_TO_VIBEVOICE_MAP.items():
            voices.append(VoiceInfo(
                voice_id=openai_name,
                name=openai_name,
                type="openai-compatible",
                gender=None
            ))

        # Add native VibeVoice voices
        for name, path in self.voice_presets.items():
            if name not in seen and "-" not in name:  # Skip full names, use short names
                path_stem = path.stem  # e.g., "en-Carter_man" or "en-Emma_woman"
                gender = "female" if "_woman" in path_stem else "male" if "_man" in path_stem else None
                voices.append(VoiceInfo(
                    voice_id=name,
                    name=name,
                    type="vibevoice-native",
                    gender=gender
                ))
                seen.add(name)

        return voices

    def _resolve_voice(self, voice: str) -> str:
        """Resolve voice name to VibeVoice voice"""
        # Check if it's an OpenAI voice name
        if voice.lower() in OPENAI_TO_VIBEVOICE_MAP:
            voice = OPENAI_TO_VIBEVOICE_MAP[voice.lower()]

        # Check if voice exists
        if voice not in self.voice_presets:
            available = [v for v in self.voice_presets.keys() if "-" not in v]
            print(f"[warning] Voice '{voice}' not found, using 'Carter'. Available: {available}")
            voice = "Carter"

        return voice

    def _get_voice_prompt(self, voice: str) -> Any:
        """Load or get cached voice prompt"""
        if voice not in self._voice_cache:
            voice_path = self.voice_presets[voice]
            print(f"[tts] Loading voice prompt from {voice_path}")
            self._voice_cache[voice] = torch.load(
                voice_path,
                map_location=self._torch_device,
                weights_only=False
            )
        return self._voice_cache[voice]

    def _prepare_inputs(self, text: str, prefilled_outputs: Any) -> Dict[str, Any]:
        """Prepare model inputs and move tensors to the selected device."""
        if not self.processor:
            raise RuntimeError("Processor not loaded")

        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self._torch_device)

        return inputs

    def _sync_device_for_timing(self) -> None:
        """Synchronize async accelerators so logged timing is meaningful."""
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _audio_to_numpy(self, audio: Any) -> np.ndarray:
        """Convert model audio output or chunk to mono float32 numpy."""
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().to(torch.float32).numpy()
        else:
            audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.reshape(-1)

        if audio.size:
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak

        return audio.astype(np.float32, copy=False)

    def _run_stream_generation(
        self,
        inputs: Dict[str, Any],
        audio_streamer: AudioStreamer,
        errors: List[BaseException],
        cfg_scale: float,
        prefilled_outputs: Any,
        stop_event: threading.Event,
    ) -> None:
        """Run generation in a background thread and feed AudioStreamer."""
        try:
            progress_enabled = os.environ.get("VIBEVOICE_PROGRESS", "").lower() in {"1", "true", "yes"}
            stderr_sink = nullcontext()
            if not progress_enabled:
                stderr_sink = open(os.devnull, "w", encoding="utf-8")

            with stderr_sink as sink:
                progress_context = nullcontext() if progress_enabled else redirect_stderr(sink)
                with progress_context, torch.inference_mode():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={"do_sample": False},
                        verbose=False,
                        audio_streamer=audio_streamer,
                        stop_check_fn=stop_event.is_set,
                        refresh_negative=True,
                        all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                    )
        except GeneratorExit:
            raise
        except BaseException as exc:
            errors.append(exc)
            traceback.print_exc()
        finally:
            audio_streamer.end()

    def _run_full_generation(
        self,
        inputs: Dict[str, Any],
        cfg_scale: float,
        prefilled_outputs: Any,
    ) -> Any:
        """Run non-streaming generation with optional progress-bar suppression."""
        progress_enabled = os.environ.get("VIBEVOICE_PROGRESS", "").lower() in {"1", "true", "yes"}
        stderr_sink = nullcontext()
        if not progress_enabled:
            stderr_sink = open(os.devnull, "w", encoding="utf-8")

        with stderr_sink as sink:
            progress_context = nullcontext() if progress_enabled else redirect_stderr(sink)
            with progress_context, torch.inference_mode():
                return self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                )

    def generate_speech(self, text: str, voice: str, cfg_scale: float = 1.5) -> np.ndarray:
        """Generate speech from text

        Args:
            text: Text to synthesize
            voice: Voice name
            cfg_scale: CFG scale for generation

        Returns:
            Audio samples as numpy array (float32, 24kHz)
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        voice = self._resolve_voice(voice)
        prefilled_outputs = self._get_voice_prompt(voice)

        # Clean text
        text = text.strip().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        inputs = self._prepare_inputs(text, prefilled_outputs)

        print(f"[tts] Generating speech for {len(text)} chars with voice '{voice}'")
        start_time = time.time()

        # Generate
        outputs = self._run_full_generation(
            inputs=inputs,
            cfg_scale=cfg_scale,
            prefilled_outputs=prefilled_outputs,
        )

        self._sync_device_for_timing()
        elapsed = time.time() - start_time

        # Extract audio
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = self._audio_to_numpy(outputs.speech_outputs[0])

            duration = len(audio) / SAMPLE_RATE
            rtf = elapsed / duration if duration > 0 else float("inf")
            print(f"[tts] Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f}x)")

            return audio
        else:
            raise RuntimeError("No audio output generated")

    def stream_speech_pcm(self, text: str, voice: str, cfg_scale: float = 1.5, volume: float = 1.0) -> Iterator[bytes]:
        """Generate speech and yield raw PCM16 chunks as soon as they are available."""
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        voice = self._resolve_voice(voice)
        prefilled_outputs = self._get_voice_prompt(voice)
        text = text.strip().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        inputs = self._prepare_inputs(text, prefilled_outputs)

        print(f"[tts] Streaming speech for {len(text)} chars with voice '{voice}'")
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        stop_event = threading.Event()
        errors: List[BaseException] = []
        start_time = time.time()
        first_chunk_logged = False

        thread = threading.Thread(
            target=self._run_stream_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        thread.start()

        try:
            for audio_chunk in audio_streamer.get_stream(0):
                audio = self._audio_to_numpy(audio_chunk)
                if audio.size == 0:
                    continue

                if not first_chunk_logged:
                    first_chunk_logged = True
                    first_chunk_ms = (time.time() - start_time) * 1000
                    print(f"[tts] First PCM chunk ready in {first_chunk_ms:.0f} ms")

                yield audio_to_pcm16(audio, volume=volume)

            if errors:
                raise RuntimeError(str(errors[0]))
        finally:
            stop_event.set()
            audio_streamer.end()
            if thread is not threading.current_thread():
                thread.join(timeout=5)


# ------------------------------------------------------------------------------
# Audio Format Conversion
# ------------------------------------------------------------------------------

def convert_audio(audio: np.ndarray, format: str, sample_rate: int = SAMPLE_RATE, volume: float = 1.0) -> bytes:
    """Convert audio to specified format using ffmpeg

    Args:
        audio: Audio samples (float32, mono)
        format: Output format (mp3, wav, opus, flac, aac, pcm)
        sample_rate: Sample rate

    Returns:
        Audio bytes in specified format
    """
    format = format.lower()

    if format == "pcm":
        # Raw PCM16 little-endian
        return audio_to_pcm16(audio, volume=volume)

    if format == "wav":
        # Use scipy for WAV
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_to_pcm16_array(audio, volume=volume))
        return buffer.getvalue()

    # Use ffmpeg for other formats
    # Prepare input WAV
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, audio_to_pcm16_array(audio, volume=volume))
    wav_data = wav_buffer.getvalue()

    if format not in FFMPEG_FORMAT_ARGS:
        raise ValueError(f"Unsupported format: {format}")

    # Run ffmpeg
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "wav",
        "-i", "pipe:0",
        *FFMPEG_FORMAT_ARGS[format],
        "pipe:1"
    ]

    try:
        result = subprocess.run(
            cmd,
            input=wav_data,
            capture_output=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[error] ffmpeg failed: {e.stderr.decode()}")
        raise RuntimeError(f"Audio conversion failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")


def stream_encoded_audio(pcm_chunks: Iterator[bytes], format: str) -> Iterator[bytes]:
    """Encode a PCM16 chunk stream with ffmpeg and yield encoded audio chunks."""
    format = format.lower()
    if format == "pcm":
        yield from pcm_chunks
        return

    if format not in FFMPEG_FORMAT_ARGS:
        raise ValueError(f"Streaming is not supported for format: {format}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-i", "pipe:0",
        *FFMPEG_FORMAT_ARGS[format],
        "pipe:1",
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    writer_errors: List[BaseException] = []

    def write_input() -> None:
        try:
            assert process.stdin is not None
            for chunk in pcm_chunks:
                process.stdin.write(chunk)
                process.stdin.flush()
        except BaseException as exc:
            writer_errors.append(exc)
        finally:
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass

    writer = threading.Thread(target=write_input, daemon=True)
    writer.start()

    completed = False
    try:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(16 * 1024)
            if not chunk:
                completed = True
                break
            yield chunk
    finally:
        if not completed and process.poll() is None:
            process.terminate()
        writer.join(timeout=5)

    return_code = process.wait()
    stderr = process.stderr.read() if process.stderr else b""
    if writer_errors:
        raise RuntimeError(str(writer_errors[0]))
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")


def audio_to_pcm16_array(audio: np.ndarray, volume: float = 1.0) -> np.ndarray:
    """Convert float32 mono audio samples to clipped PCM16 samples."""
    safe_volume = max(0.0, min(float(volume), 4.0))
    return (np.clip(audio * safe_volume, -1.0, 1.0) * 32767).astype(np.int16)


def audio_to_pcm16(audio: np.ndarray, volume: float = 1.0) -> bytes:
    """Convert float32 mono audio samples to raw little-endian PCM16 bytes."""
    return audio_to_pcm16_array(audio, volume=volume).tobytes()


def resolve_volume(requested_volume: Optional[float], requested_multiplier: Optional[float]) -> float:
    """Resolve request volume against env default and clamp to a safe range."""
    if requested_volume is None and requested_multiplier is not None:
        requested_volume = requested_multiplier
    elif requested_volume is None:
        requested_volume = DEFAULT_VOLUME
    return max(0.0, min(float(requested_volume), 4.0))


def get_content_type(format: str) -> str:
    """Get MIME content type for audio format"""
    types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "pcm": "audio/pcm",
    }
    return types.get(format.lower(), "application/octet-stream")


# ------------------------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------------------------

# Global service instance
tts_service: Optional[VibeVoiceTTSService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown"""
    global tts_service

    # --- Startup ---
    model_path = os.environ.get("VIBEVOICE_MODEL_PATH", DEFAULT_MODEL_PATH)
    device = os.environ.get("VIBEVOICE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    tts_service = VibeVoiceTTSService(model_path=model_path, device=device)
    try:
        tts_service.load()
    except Exception as e:
        print(f"[FATAL] Model loading failed: {e}")
        traceback.print_exc()

    yield

    # --- Shutdown ---
    if tts_service and tts_service.model:
        del tts_service.model
        torch.cuda.empty_cache()


app = FastAPI(
    title="VibeVoice TTS Server",
    description="OpenAI-compatible TTS API powered by VibeVoice-Realtime-0.5B",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        service="vibevoice-realtime-openai-api",
        model_loaded=tts_service is not None and tts_service.model is not None,
        device=tts_service.device if tts_service else "unknown",
        features={
            "streaming": True,
            "streaming_formats": STREAMABLE_FORMATS,
            "formats": SUPPORTED_FORMATS,
            "sample_rate": SAMPLE_RATE,
            "default_volume": DEFAULT_VOLUME,
        }
    )


@app.get("/v1/audio/voices", response_model=VoicesResponse)
async def list_voices():
    """List available voices (OpenAI-compatible)"""
    if not tts_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    return VoicesResponse(voices=tts_service.get_available_voices())


@app.get("/v1/audio/models")
async def list_models():
    """List available TTS models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1699000000,
                "owned_by": "vibevoice",
                "name": "VibeVoice-Realtime-0.5B"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1699000000,
                "owned_by": "vibevoice",
                "name": "VibeVoice-Realtime-0.5B"
            }
        ]
    }


@app.post("/v1/audio/speech")
def create_speech(request: TTSRequest):
    """Generate speech from text (OpenAI-compatible)"""
    if not tts_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Validate input
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    if len(request.input) > 4096:
        raise HTTPException(status_code=400, detail="Input text exceeds 4096 characters")

    response_format = request.response_format.lower()
    volume = resolve_volume(request.volume, request.volume_multiplier)

    if response_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {SUPPORTED_FORMATS}"
        )

    if request.stream and response_format not in STREAMABLE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Streaming is currently supported only with: {STREAMABLE_FORMATS}"
        )

    try:
        if request.stream or response_format in STREAMABLE_FORMATS:
            if response_format in FFMPEG_FORMAT_ARGS and shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

            pcm_chunks = tts_service.stream_speech_pcm(
                text=request.input,
                voice=request.voice,
                cfg_scale=CFG_SCALE,
                volume=volume,
            )
            audio_chunks = stream_encoded_audio(pcm_chunks, response_format)
            return StreamingResponse(
                audio_chunks,
                media_type=get_content_type(response_format),
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # Generate speech
        audio = tts_service.generate_speech(
            text=request.input,
            voice=request.voice,
            cfg_scale=CFG_SCALE,
        )

        # Convert to requested format
        audio_bytes = convert_audio(audio, response_format, volume=volume)
        content_type = get_content_type(response_format)

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{response_format}"
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VibeVoice OpenAI-Compatible TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8880, help="Port to bind")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Device")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Set environment variables for startup
    os.environ["VIBEVOICE_MODEL_PATH"] = args.model_path
    os.environ["VIBEVOICE_DEVICE"] = args.device

    print(f"Starting VibeVoice TTS Server on http://{args.host}:{args.port}")
    print(f"OpenAI TTS endpoint: http://{args.host}:{args.port}/v1/audio/speech")

    uvicorn.run(
        "vibevoice_realtime_openai_api:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    # To suppress warnings, run with: python -W ignore vibevoice_realtime_openai_api.py
    main()
