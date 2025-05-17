# hubconf.py for local model loading in RealtimeSTT/silero_assets
import os
import torch
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Union, Tuple

# Ensure this is at the top, before any utility function definitions
# _PACKAGE_DIR will point to the 'silero_assets' directory where this hubconf.py resides
_PACKAGE_DIR = Path(__file__).parent

DEPENDENCIES = ['torch', 'numpy']

# Silero VAD models are versioned.
# RealtimeSTT defaults to 'v4'. The files you have ('silero_vad.jit', 'silero_vad.onnx')
# are the larger, original models (let's call them 'v3' style for consistency with hubconf).
# This hubconf will serve these files even if 'v4' is requested for 'silero_vad'.
_DEFAULT_VERSION = 'v4' # Aligns with RealtimeSTT's default if no version is passed by it.

_INTERNALS_MODELS = {
    'v3': {
        'silero_vad.jit': 'silero_vad.jit',
        'silero_vad.onnx': 'silero_vad.onnx',
        'silero_vad_micro.jit': 'silero_vad_micro.jit',
        'silero_vad_micro.onnx': 'silero_vad_micro.onnx',
        'silero_vad_quantized.jit': 'silero_vad_quantized.jit', # Example, if you had this
    },
    'v4': {
        # In original Silero repo, v4 'silero_vad.jit' points to a smaller model 'v4/silero_vad.jit'.
        # We are overriding this to point to your local 'silero_vad.jit' (the larger one).
        'silero_vad.jit': 'silero_vad.jit',
        'silero_vad.onnx': 'silero_vad.onnx',
        'silero_vad_micro.jit': 'silero_vad_micro.jit', # If you had micro_v4, it would go here
        'silero_vad_micro.onnx': 'silero_vad_micro.onnx',
    }
}


def _download_internals(name: str, version: str = None, force_reload: bool = False):
    """
    Modified to load models directly from the _PACKAGE_DIR (silero_assets).
    'name' is the model key like 'silero_vad.jit'.
    'version' is requested model version, e.g., 'v4'.
    """
    version = version or _DEFAULT_VERSION # Ensure a version is always selected

    if version not in _INTERNALS_MODELS:
        raise ValueError(f"Version {version} not supported by this local hubconf.")
    if name not in _INTERNALS_MODELS[version]:
        raise ValueError(f"Model name {name} for version {version} not supported by this local hubconf.")

    actual_filename = _INTERNALS_MODELS[version][name]

    # **** Key modification for your setup ****
    # If the requested model is 'silero_vad.jit' or 'silero_vad.onnx',
    # always serve the files you have at the root of 'silero_assets', regardless of 'version'.
    if name == 'silero_vad.jit':
        actual_filename = 'silero_vad.jit'
    elif name == 'silero_vad.onnx':
        actual_filename = 'silero_vad.onnx'
    # For other models like 'silero_vad_micro.jit', it will use the mapped 'actual_filename'.
    # If that mapped filename (e.g., 'silero_vad_micro.jit') doesn't exist in 'silero_assets', it will fail.

    path = _PACKAGE_DIR / actual_filename
    
    if not path.exists():
        error_message = (
            f"Model file '{path}' (derived from request: name='{name}', version='{version}') not found "
            f"in '{_PACKAGE_DIR}'. Ensure it is correctly placed in the 'silero_assets' directory. "
            f"Currently, only 'silero_vad.jit' and 'silero_vad.onnx' are assumed to be present at the root."
        )
        if "micro" in name: # More specific error for micro versions
            error_message += (
                f" If you need a 'micro' VAD model, please download the appropriate "
                f"'.jit' or '.onnx' file (e.g., 'silero_vad_micro.jit') and place it in '{_PACKAGE_DIR}'."
            )
        raise FileNotFoundError(error_message)
    return str(path)


# ======= UTILITY FUNCTIONS (Copied from snakers4/silero-vad/hubconf.py) ========

class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30,
                 return_seconds: bool = False,
                 progress_tracking_callback: Callable[[float], None] = None
                 ):

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.return_seconds = return_seconds
        self.progress_tracking_callback = progress_tracking_callback

        if min_silence_duration_ms < 0:
            raise ValueError('min_silence_duration_ms must be positive')
        if speech_pad_ms < 0:
            raise ValueError('speech_pad_ms must be positive')
        if sampling_rate != 16000 and sampling_rate != 8000:
            raise ValueError('sampling_rate must be 16000 or 8000')

        self.reset_states() # call reset_states from __init__
        self.sample_to_ms = 1000 / sampling_rate

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=None):
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio data must be preferably a torch tensor or convertible to it")

        if len(x.shape) == 1:  # mono
            x = x.unsqueeze(0)
        if len(x.shape) > 2:
            raise ValueError(f"Too many dimensions for input audio tensor {x.shape}")
        if x.shape[0] > 1:
            raise ValueError("Main VAD model is mono, for multi-channel input use vad_fast") # or manually iterate over channels
        if x.dtype != torch.float32: # May harm performance
            x = x.to(torch.float32)

        window_size_samples = x.shape[-1] # Use actual size of the input chunk
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - window_size_samples - self.speech_pad_ms * self.sample_to_ms
            if speech_start < 0:
                speech_start = 0
            return {'start': int(speech_start) if not (return_seconds or self.return_seconds) else round(speech_start / self.sampling_rate, 3)}


        if (speech_prob < self.threshold - 0.15) and self.triggered: # Refined threshold condition
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end >= self.min_silence_duration_ms * self.sample_to_ms :
                speech_end = self.temp_end + self.speech_pad_ms * self.sample_to_ms
                if speech_end > self.current_sample:
                     speech_end = self.current_sample
                self.triggered = False
                self.temp_end = 0
                return {'end': int(speech_end) if not (return_seconds or self.return_seconds) else round(speech_end / self.sampling_rate, 3)}


        if self.progress_tracking_callback is not None:  # Ensure callback is defined
            # Example: progress assuming a fixed total duration like 60 seconds of processing
            # This part might need adjustment based on how total duration is known or estimated
            # For continuous streams, this might represent chunks processed / total expected chunks or similar
            # A simple placeholder might be:
            # self.progress_tracking_callback(self.current_sample / (self.sampling_rate * 60)) # progress in minutes processed
            # If total duration is unknown, a chunk count or processed time could be passed.
            # This callback's design depends heavily on the application's known parameters.
            pass # Placeholder for progress tracking logic

        return None


def get_speech_timestamps(audio: torch.Tensor,
                          model,
                          threshold: float = 0.5,
                          sampling_rate: int = 16000,
                          min_silence_duration_ms: int = 100,
                          speech_pad_ms: int = 30,
                          min_speech_duration_ms: int = 250,
                          return_seconds: bool = False,
                          progress_tracking_callback: Callable[[float], None] = None
                          ):

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio data must be preferably a torch tensor or convertible to it")

    if len(audio.shape) > 1: # VAD model expects mono audio
        if audio.shape[0] == 1: # If shape is (1, N), squeeze it
            audio = audio.squeeze(0)
        else: # If shape is (C, N) with C > 1, or (N, C) - this needs to be handled or raise error
            raise ValueError("Audio data must be a mono tensor or squeezable to mono.")


    if sampling_rate != 16000 and sampling_rate != 8000:
        raise ValueError('sampling_rate must be 16000 or 8000')

    if min_silence_duration_ms < 0:
        raise ValueError('min_silence_duration_ms must be positive')
    if speech_pad_ms < 0:
        raise ValueError('speech_pad_ms must be positive')
    if min_speech_duration_ms < 0:
        raise ValueError('min_speech_duration_ms must be positive')

    model.reset_states() 

    # samples per frame / chunk (should align with model's expectation)
    # Common window sizes for Silero VAD based on its examples
    if sampling_rate == 16000:
        # Original Silero examples use 512, 1024, 1536 for 16kHz
        # For get_speech_timestamps, the model is fed window_size_samples at a time
        # These specific numbers are critical for models trained on fixed chunk sizes
        window_size_samples_list = [512, 1024, 1536] 
        # RealtimeSTT seems to use 512 by default for 16kHz when calling this.
        # Let's stick to a common one that should allow the model to function, but be aware it might be sensitive.
        # The underlying model.py in silero-vad handles various chunk sizes.
        # The critical part is that the model (`model(chunk, sampling_rate)`) can process it.
        # A common approach is to just pick one if not specified or make it a parameter if critical.
        # Let's use 512 as a base but acknowledge the model itself might have its own internal chunking.
        window_size_samples = 512 
    elif sampling_rate == 8000:
        window_size_samples_list = [256, 512, 768]
        window_size_samples = 256
    else:
        raise ValueError(f"Unsupported sampling rate: {sampling_rate}. Must be 8000 or 16000.")

    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15  # to avoid being too sensitive
    triggered = False # found speech
    
    temp_end = 0 # loop over audio
    len_audio = len(audio)

    # Pre-calculate some constants
    ms_to_samples_factor = sampling_rate / 1000
    min_silence_samples = min_silence_duration_ms * ms_to_samples_factor
    speech_pad_samples = speech_pad_ms * ms_to_samples_factor
    min_speech_samples = min_speech_duration_ms * ms_to_samples_factor

    for current_sample_idx in range(0, len_audio, window_size_samples):
        chunk = audio[current_sample_idx : current_sample_idx + window_size_samples]
        
        if len(chunk) < window_size_samples: # If last chunk is smaller, pad with zeros
            chunk = torch.cat([chunk, torch.zeros(window_size_samples - len(chunk))])

        speech_prob = model(chunk.unsqueeze(0), sampling_rate).item() # model expects batch dim

        # Simplified logic:
        if speech_prob >= threshold and not triggered: # Start of speech
            triggered = True
            current_speech['start'] = int(max(0, current_sample_idx - speech_pad_samples))
            # No 'end' yet, speech is ongoing

        elif speech_prob < neg_threshold and triggered: # Potential end of speech
            if temp_end == 0: # First time we see silence after speech
                temp_end = current_sample_idx + len(chunk) # Mark end of current chunk as potential speech end

            # Check if silence duration is met since temp_end
            # current_sample_idx + len(chunk) is the *end* of the current processing window
            if (current_sample_idx + len(chunk)) - temp_end >= min_silence_samples:
                current_speech['end'] = int(min(len_audio, temp_end + speech_pad_samples))
                
                # Check minimum speech duration
                if (current_speech['end'] - current_speech['start']) >= min_speech_samples:
                    speeches.append(current_speech)

                current_speech = {}
                triggered = False
                temp_end = 0
        
        elif speech_prob >= threshold and triggered: # Speech continues
            temp_end = 0 # Reset temp_end as speech is ongoing

        if progress_tracking_callback and (current_sample_idx // window_size_samples) % 20 == 0:
             progress = (current_sample_idx + len(chunk)) / len_audio
             progress_tracking_callback(progress)

    # If loop finishes and speech was ongoing
    if triggered and 'start' in current_speech:
        current_speech['end'] = int(min(len_audio, len_audio + speech_pad_samples)) # Ends at audio length
        if (current_speech['end'] - current_speech['start']) >= min_speech_samples:
            speeches.append(current_speech)

    if return_seconds:
        for speech_dict in speeches:
            speech_dict['start'] = round(speech_dict['start'] / sampling_rate, 3)
            speech_dict['end'] = round(speech_dict['end'] / sampling_rate, 3)
    
    return speeches


class Validator: # Validates model inputs, raises exceptions otherwise
    def __init__(self, url_or_path, force_reload):
        self.url_or_path = url_or_path
        if Path(self.url_or_path).is_file() and not force_reload:
            self.local_path = self.url_or_path
        else:
            # In our local setup, we don't download. This part is mostly legacy
            # from the original hubconf but kept for structure.
            # The actual path finding is done in _download_internals.
            # If _download_internals fails, this Validator won't even be called with a remote URL.
            # If it's called with a local path (from _download_internals), it should just use it.
            if Path(self.url_or_path).is_file():
                 self.local_path = self.url_or_path
            else:
                # This case should ideally not be hit if _download_internals correctly provides a local path.
                raise FileNotFoundError(f"Validator could not find file at {self.url_or_path}. "
                                        "This might indicate an issue with how hubconf is being used or "
                                        "that _download_internals did not correctly resolve the local path.")

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input type is not torch.Tensor')
        if not x.ndim == 1:
            raise ValueError('Input tensor ndim is not 1')
        if not x.dtype == torch.float32:
            # Consider warning instead of error for dtype, or ensuring conversion
            # For now, align with original which implies an expectation of float32
            print('Warning: Input tensor dtype is not float32. Silero VAD models usually expect float32.')
            # x = x.to(torch.float32) # Optionally convert
        # Device check can be tricky; model might be moved to GPU later.
        # For now, let's assume CPU data initially, or remove this check.
        # if not x.device == torch.device('cpu'):
        #     print('Warning: Input tensor device is not cpu.')
        return x

def save_audio(path: str,
               tensor: torch.Tensor,
               sampling_rate: int = 16000):
    if not Path(path).parent.exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True) # Added exist_ok=True
    if not tensor.ndim == 1: 
        raise ValueError("Audio tensor `tensor` should be 1D")
    try:
        import soundfile as sf
        sf.write(path, tensor.cpu().numpy(), sampling_rate) # Ensure tensor is on CPU for numpy conversion
    except ImportError:
        raise ImportError("Please install soundfile: 'pip install soundfile'") 
    except Exception as e:
        raise RuntimeError(f"Error saving audio: {e}")


def read_audio(path: str,
               sampling_rate: int = 16000):
    try:
        import soundfile as sf
        wav, sr = sf.read(path) # wav is a numpy array
    except ImportError:
        raise ImportError("Please install soundfile: 'pip install soundfile'")
    except Exception as e:
        raise RuntimeError(f"Error reading audio: {e}")

    if sr != sampling_rate and sampling_rate != -1: # sampling_rate == -1 means load native
        if wav.ndim > 1: # if stereo, take the first channel before resampling
            wav = wav[:, 0]
        try:
            import torchaudio
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0) # Add channel dim for torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
            wav_resampled = resampler(wav_tensor).squeeze(0).numpy() # Remove channel dim
            wav = wav_resampled
            sr = sampling_rate # Update sr to the new sampling rate
        except ImportError:
            raise ImportError("Please install torchaudio for resampling: 'pip install torchaudio'")
        except Exception as e:
            raise RuntimeError(f"Error resampling audio: {e}. Original SR: {sr}, Target SR: {sampling_rate}")
    
    if wav.ndim > 1: # If still multi-channel (e.g. didn't resample and was stereo), take first channel
         wav = wav[:, 0]

    return torch.from_numpy(wav).float()


def get_number_audio_chunks(audio: torch.Tensor,
                            sampling_rate: int = 16000):
    # This util gives an estimate of chunks for a given window size.
    # Window size depends on model and sampling rate.
    # Silero VAD models typically use 512 samples for 16kHz (32ms) or 256 for 8kHz.
    if sampling_rate == 16000:
        chunk_size = 512 
    elif sampling_rate == 8000:
        chunk_size = 256
    else:
        raise ValueError("Sampling rate must be 16000 or 8000 Hz")
    
    if not torch.is_tensor(audio) or audio.ndim != 1:
        raise ValueError("Input audio must be a 1D torch tensor.")
        
    return (audio.shape[0] + chunk_size - 1) // chunk_size


def collect_chunks(tss: List[Dict[str, int]],
                   wav: torch.Tensor):
    chunks = []
    for i in tss:
        chunks.append(wav[i['start']:i['end']])
    return torch.cat(chunks) if chunks else torch.empty(0) # Handle case where no speech chunks are found

# List of utility functions/classes to be returned by model loading functions
_UTILS = (get_speech_timestamps,
          save_audio,
          read_audio,
          VADIterator,
          collect_chunks,
          Validator, 
          get_number_audio_chunks)


# ======= MODEL LOADING FUNCTIONS ========

def silero_vad(onnx: bool = False, force_reload: bool = False, version: str = None, trust_repo: bool = None, trust_remote_code: bool = None): # Added trust_remote_code alias
    """
    Loads the standard Silero VAD model.
    """
    version = version or _DEFAULT_VERSION
    # Logic in _download_internals already maps RealtimeSTT's 'v4' request for 'silero_vad.jit'
    # to your root 'silero_vad.jit' file. So, we don't need strict version check here against _INTERNALS_MODELS keys
    # as long as _download_internals can find the file.

    if onnx:
        path = _download_internals(name='silero_vad.onnx', version=version, force_reload=force_reload)
        try:
            import onnxruntime # type: ignore
        except ImportError:
            raise ImportError("Please install onnxruntime: 'pip install onnxruntime'")
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        model = onnxruntime.InferenceSession(path, sess_options=opts) 
    else:
        path = _download_internals(name='silero_vad.jit', version=version, force_reload=force_reload)
        model = torch.jit.load(path) # type: ignore

    return model, _UTILS


def silero_vad_micro(onnx: bool = False, force_reload: bool = False, version: str = None, trust_repo: bool = None, trust_remote_code: bool = None): # Added trust_remote_code alias
    """
    Loads the smaller "micro" Silero VAD model.
    Note: You will need to have 'silero_vad_micro.jit' or 'silero_vad_micro.onnx'
          in your 'silero_assets' folder for this to work locally.
    """
    version = version or _DEFAULT_VERSION

    if onnx:
        path = _download_internals(name='silero_vad_micro.onnx', version=version, force_reload=force_reload)
        try:
            import onnxruntime # type: ignore
        except ImportError:
            raise ImportError("Please install onnxruntime: 'pip install onnxruntime'")
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        model = onnxruntime.InferenceSession(path, sess_options=opts)
    else:
        path = _download_internals(name='silero_vad_micro.jit', version=version, force_reload=force_reload)
        model = torch.jit.load(path) # type: ignore

    return model, _UTILS


# Example: silero_vad_quantized (if you were to add it and its files)
# def silero_vad_quantized(force_reload: bool = False, version: str = None, trust_repo: bool = None, trust_remote_code: bool = None):
#     """
#     Loads a quantized Silero VAD model (JIT only typically).
#     Note: You will need 'silero_vad_quantized.jit' in 'silero_assets'.
#     """
#     version = version or _DEFAULT_VERSION 
#     # The original quantized model was not typically versioned like v3/v4,
#     # so _download_internals needs to have a mapping for 'silero_vad_quantized.jit'
#     # under the version you pass (e.g., 'v3' or 'v4' if you align it that way).
#     # For simplicity, let's assume if you use this, you've added 'silero_vad_quantized.jit'
#     # to _INTERNALS_MODELS['v3'] or ['v4'] and placed the file.
#     
#     # Force a specific internal name if it's always the same regardless of version argument
#     quantized_model_name_in_map = 'silero_vad_quantized.jit' # This must exist in _INTERNALS_MODELS for the 'version'
#
#     path = _download_internals(name=quantized_model_name_in_map, version=version, force_reload=force_reload)
#     model = torch.jit.load(path)
#     return model, _UTILS
