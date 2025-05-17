"""
 The AudioToTextRecorder class in the provided code facilitates fast speech-to-text transcription.
 The class employs the faster_whisper library to transcribe the recorded audio into text using machine learning models, which can be run either on a GPU or CPU.
 Voice activity detection (VAD) is built in, meaning the software can automatically start or stop recording based on the presence or absence of speech.
 It integrates wake word detection through the pvporcupine library, allowing the software to initiate recording when a specific word or phrase is spoken.
 The system provides real-time feedback and can be further customized.

Features:
- Voice Activity Detection: Automatically starts/stops recording when speech
  is detected or when speech ends.
- Wake Word Detection: Starts recording when a specified wake word (or words)
  is detected.
- Event Callbacks: Customizable callbacks for when recording starts
  or finishes.
- Fast Transcription: Returns the transcribed text from the audio as fast
  as possible.

Author: Kolja Beigel

"""

from faster_whisper import WhisperModel, BatchedInferencePipeline
from typing import Iterable, List, Optional, Union
from openwakeword.model import Model
import torch.multiprocessing as mp
from scipy.signal import resample
import signal as system_signal
from ctypes import c_bool
from scipy import signal
from .safepipe import SafePipe
import soundfile as sf
import faster_whisper
import openwakeword
import collections
import numpy as np
import pvporcupine
import traceback
import threading
import webrtcvad
import datetime
import platform
import logging
import struct
import base64
import queue
import torch
import halo
import time
import copy
import os
import re
import gc

# Named logger for this module.
logger = logging.getLogger("realtimestt")
logger.propagate = False

# Set OpenMP runtime duplicate library handling to OK (Use only for development!)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_MODEL_TRANSCRIPTION_REALTIME = "tiny"
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_REALTIME_INITIAL_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0
INIT_WAKE_WORD_BUFFER_DURATION = 0.1
ALLOWED_LATENCY_LIMIT = 100

TIME_SLEEP = 0.02
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True


class TranscriptionWorker:
    def __init__(self, conn, stdout_pipe, model_path, download_root, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens,
                 batch_size, faster_whisper_vad_filter, normalize_audio):
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.model_path = model_path
        self.download_root = download_root
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.batch_size = batch_size
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.normalize_audio = normalize_audio
        self.queue = queue.Queue()

    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def poll_connection(self):
        while not self.shutdown_event.is_set():
            try:
                if self.conn.poll(0.01): 
                    data = self.conn.recv()
                    self.queue.put(data)
                else:
                    time.sleep(TIME_SLEEP)
            except Exception as e:
                logging.error(f"Error receiving data from connection: {e}", exc_info=True)
                time.sleep(TIME_SLEEP)

    def run(self):
        if __name__ == "__main__":
             system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
             __builtins__['print'] = self.custom_print

        logging.info(f"Initializing faster_whisper main transcription model {self.model_path}")

        try:
            model = faster_whisper.WhisperModel(
                model_size_or_path=self.model_path,
                device=self.device,
                compute_type=self.compute_type,
                device_index=self.gpu_device_index,
                download_root=self.download_root,
            )
            if self.batch_size > 0:
                model = BatchedInferencePipeline(model=model)

            current_dir = os.path.dirname(os.path.realpath(__file__))
            warmup_audio_path = os.path.join(
                current_dir, "warmup_audio.wav"
            )
            warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
            segments, info = model.transcribe(warmup_audio_data, language="en", beam_size=1)
            model_warmup_transcription = " ".join(segment.text for segment in segments)
        except Exception as e:
            logging.exception(f"Error initializing main faster_whisper transcription model: {e}")
            raise

        self.ready_event.set()
        logging.debug("Faster_whisper main speech to text transcription model initialized successfully")

        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language, use_prompt = self.queue.get(timeout=0.1)
                    try:
                        logging.debug(f"Transcribing audio with language {language}")
                        start_t = time.time()

                        if audio is not None and audio .size > 0:
                            if self.normalize_audio:
                                peak = np.max(np.abs(audio))
                                if peak > 0:
                                    audio = (audio / peak) * 0.95
                        else:
                            logging.error("Received None audio for transcription")
                            self.conn.send(('error', "Received None audio for transcription"))
                            continue

                        prompt = None
                        if use_prompt:
                            prompt = self.initial_prompt if self.initial_prompt else None

                        if self.batch_size > 0:
                            segments, info = model.transcribe(
                                audio,
                                language=language if language else None,
                                beam_size=self.beam_size,
                                initial_prompt=prompt,
                                suppress_tokens=self.suppress_tokens,
                                batch_size=self.batch_size,
                                 vad_filter=self.faster_whisper_vad_filter
                            )
                        else:
                            segments, info = model.transcribe(
                                audio,
                                language=language if language else None,
                                beam_size=self.beam_size,
                                initial_prompt=prompt,
                                suppress_tokens=self.suppress_tokens,
                                vad_filter=self.faster_whisper_vad_filter
                            )
                        elapsed = time.time() - start_t
                        transcription = " ".join(seg.text for seg in segments).strip()
                        logging.debug(f"Final text detected with main model: {transcription} in {elapsed:.4f}s")
                        self.conn.send(('success', (transcription, info)))
                    except Exception as e:
                        logging.error(f"General error in transcription: {e}", exc_info=True)
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    logging.debug("Transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    logging.error(f"General error in processing queue item: {e}", exc_info=True)
        finally:
            __builtins__['print'] = print 
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set() 
            polling_thread.join() 


class bcolors:
    OKGREEN = '\033[92m' 
    WARNING = '\033[93m' 
    ENDC = '\033[0m'     


class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting
    voice activity, and then transcribing the captured audio using the
    `faster_whisper` model.
    """

    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 download_root: str = None, 
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cuda",
                 on_recording_start=None,
                 on_recording_stop=None,
                 on_transcription_start=None,
                 ensure_sentence_starting_uppercase=True,
                 ensure_sentence_ends_with_period=True,
                 use_microphone=True,
                 spinner=True,
                 level=logging.WARNING,
                 batch_size: int = 16, 
                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 use_main_model_for_realtime=False,
                 realtime_model_type=INIT_MODEL_TRANSCRIPTION_REALTIME,
                 realtime_processing_pause=INIT_REALTIME_PROCESSING_PAUSE,
                 init_realtime_after_seconds=INIT_REALTIME_INITIAL_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,
                 realtime_batch_size: int = 16, 
                 # Voice activation parameters
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 silero_use_onnx: bool = False,
                 silero_deactivity_detection: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = (
                     INIT_POST_SPEECH_SILENCE_DURATION
                 ),
                 min_length_of_recording: float = (
                     INIT_MIN_LENGTH_OF_RECORDING
                 ),
                 min_gap_between_recordings: float = (
                     INIT_MIN_GAP_BETWEEN_RECORDINGS
                 ),
                 pre_recording_buffer_duration: float = (
                     INIT_PRE_RECORDING_BUFFER_DURATION
                 ),
                 on_vad_start=None,
                 on_vad_stop=None,
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,
                 on_turn_detection_start=None,
                 on_turn_detection_stop=None, 
                 # Wake word parameters
                 wakeword_backend: str = "",
                 openwakeword_model_paths: str = None,
                 openwakeword_inference_framework: str = "onnx",
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = (
                    INIT_WAKE_WORD_ACTIVATION_DELAY
                 ),
                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
                 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
                 on_wakeword_detected=None,
                 on_wakeword_timeout=None,
                 on_wakeword_detection_start=None,
                 on_wakeword_detection_end=None,
                 on_recorded_chunk=None,
                 debug_mode=False,
                 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
                 beam_size: int = 5,
                 beam_size_realtime: int = 3,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 initial_prompt: Optional[Union[str, Iterable[int]]] = None,
                 initial_prompt_realtime: Optional[Union[str, Iterable[int]]] = None,
                 suppress_tokens: Optional[List[int]] = [-1],
                 print_transcription_time: bool = False,
                 early_transcription_on_silence: int = 0,
                 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT,
                 no_log_file: bool = False,
                 use_extended_logging: bool = False,
                 faster_whisper_vad_filter: bool = True,
                 normalize_audio: bool = False,
                 start_callback_in_new_thread: bool = False,
                 ):
        """ Docstring as provided by user ... """
        
        # Using 8 spaces for these direct assignments, as in original code
        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.wake_words = wake_words
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.ensure_sentence_starting_uppercase = (
            ensure_sentence_starting_uppercase
        )
        self.ensure_sentence_ends_with_period = (
            ensure_sentence_ends_with_period
        )
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_start = on_vad_start
        self.on_vad_stop = on_vad_stop
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_turn_detection_start = on_turn_detection_start
        self.on_turn_detection_stop = on_turn_detection_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.on_transcription_start = on_transcription_start
        self.enable_realtime_transcription = enable_realtime_transcription
        self.use_main_model_for_realtime = use_main_model_for_realtime
        self.main_model_type = model
        if not download_root:
            download_root = None
        self.download_root = download_root
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.init_realtime_after_seconds = init_realtime_after_seconds
        self.on_realtime_transcription_update = (
            on_realtime_transcription_update
        )
        self.on_realtime_transcription_stabilized = (
            on_realtime_transcription_stabilized
        )
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.allowed_latency_limit = allowed_latency_limit
        self.batch_size = batch_size
        self.realtime_batch_size = realtime_batch_size

        self.level = level
        self.audio_queue = mp.Queue()
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.last_recording_start_time = 0
        self.last_recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.speech_end_silence_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.silero_deactivity_detection = silero_deactivity_detection
        self.silero_use_onnx = silero_use_onnx 
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.stream = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.backdate_stop_seconds = 0.0
        self.backdate_resume_seconds = 0.0
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.initial_prompt = initial_prompt
        self.initial_prompt_realtime = initial_prompt_realtime
        self.suppress_tokens = suppress_tokens
        self.use_wake_words = wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}
        self.detected_language = None
        self.detected_language_probability = 0
        self.detected_realtime_language = None
        self.detected_realtime_language_probability = 0
        self.transcription_lock = threading.Lock()
        self.shutdown_lock = threading.Lock()
        self.transcribe_count = 0
        self.print_transcription_time = print_transcription_time
        self.early_transcription_on_silence = early_transcription_on_silence
        self.use_extended_logging = use_extended_logging
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.normalize_audio = normalize_audio
        self.awaiting_speech_end = False
        self.start_callback_in_new_thread = start_callback_in_new_thread

        # ----------------------------------------------------------------------------
        # Named logger configuration (8-space indent for this block)
        logger.setLevel(logging.DEBUG) 

        log_format = "RealTimeSTT: %(name)s - %(levelname)s - %(message)s"
        file_log_format = "%(asctime)s.%(msecs)03d - " + log_format

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(logging.Formatter(log_format))

        logger.addHandler(console_handler)

        if not no_log_file:
            file_handler = logging.FileHandler('realtimesst.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(file_log_format, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(file_handler)
        # ----------------------------------------------------------------------------

        self.is_shut_down = False
        self.shutdown_event = mp.Event()

        try:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            logger.info(f"Start method has already been set. Details: {e}")

        logger.info("Starting RealTimeSTT")

        if use_extended_logging:
            logger.info("RealtimeSTT was called with these parameters:")
            # Using 12 spaces for items in this loop, as in original
            for param, value in locals().items():
                if param == "self": continue
                logger.info(f"            {param}: {value}")

        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event()

        self.parent_transcription_pipe, child_transcription_pipe = SafePipe()
        self.parent_stdout_pipe, child_stdout_pipe = SafePipe()

        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

        self.transcript_process = self._start_thread(
            target=AudioToTextRecorder._transcription_worker,
            args=(
                child_transcription_pipe,
                child_stdout_pipe,
                self.main_model_type,
                self.download_root,
                self.compute_type,
                self.gpu_device_index,
                self.device,
                self.main_transcription_ready_event,
                self.shutdown_event,
                self.interrupt_stop_event,
                self.beam_size,
                self.initial_prompt,
                self.suppress_tokens,
                self.batch_size,
                self.faster_whisper_vad_filter,
                self.normalize_audio,
            )
        )

        if self.use_microphone.value:
            logger.info("Initializing audio recording"
                         " (creating pyAudio input stream,"
                         f" sample rate: {self.sample_rate}"
                         f" buffer size: {self.buffer_size}"
                         )
            self.reader_process = self._start_thread(
                target=AudioToTextRecorder._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.use_microphone
                )
            )

        if self.enable_realtime_transcription and not self.use_main_model_for_realtime:
            try:
                logger.info("Initializing faster_whisper realtime "
                             f"transcription model {self.realtime_model_type}, "
                             f"default device: {self.device}, "
                             f"compute type: {self.compute_type}, "
                             f"device index: {self.gpu_device_index}, "
                             f"download root: {self.download_root}"
                             )
                self.realtime_model_type = faster_whisper.WhisperModel(
                    model_size_or_path=self.realtime_model_type,
                    device=self.device,
                    compute_type=self.compute_type,
                    device_index=self.gpu_device_index,
                    download_root=self.download_root,
                )
                if self.realtime_batch_size > 0:
                    self.realtime_model_type = BatchedInferencePipeline(model=self.realtime_model_type)

                current_dir = os.path.dirname(os.path.realpath(__file__))
                warmup_audio_path = os.path.join(
                    current_dir, "warmup_audio.wav"
                )
                warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
                segments, info = self.realtime_model_type.transcribe(warmup_audio_data, language="en", beam_size=1)
                model_warmup_transcription = " ".join(segment.text for segment in segments)
            except Exception as e:
                logger.exception("Error initializing faster_whisper "
                                  f"realtime transcription model: {e}"
                                  )
                raise
            logger.debug("Faster_whisper realtime speech to text "
                          "transcription model initialized successfully")

        if wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords', 'pvp', 'pvporcupine'}:
            self.wakeword_backend = wakeword_backend
            self.wake_words_list = [
                word.strip() for word in wake_words.lower().split(',')
            ]
            self.wake_words_sensitivity = wake_words_sensitivity
            self.wake_words_sensitivities = [
                float(wake_words_sensitivity)
                for _ in range(len(self.wake_words_list))
            ]

            if wake_words and self.wakeword_backend in {'pvp', 'pvporcupine'}:
                try:
                    self.porcupine = pvporcupine.create(
                        keywords=self.wake_words_list,
                        sensitivities=self.wake_words_sensitivities
                    )
                    self.buffer_size = self.porcupine.frame_length
                    self.sample_rate = self.porcupine.sample_rate
                except Exception as e:
                    logger.exception(
                        "Error initializing porcupine "
                        f"wake word detection engine: {e}. "
                        f"Wakewords: {self.wake_words_list}."
                    )
                    raise
                logger.debug(
                    "Porcupine wake word detection engine initialized successfully"
                )
            elif wake_words and self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
                openwakeword.utils.download_models()
                try:
                    if openwakeword_model_paths:
                        model_paths = openwakeword_model_paths.split(',')
                        self.owwModel = Model(
                            wakeword_models=model_paths,
                            inference_framework=openwakeword_inference_framework
                        )
                        logger.info(
                            "Successfully loaded wakeword model(s): "
                            f"{openwakeword_model_paths}"
                        )
                    else:
                        self.owwModel = Model(
                            inference_framework=openwakeword_inference_framework)
                    self.oww_n_models = len(self.owwModel.models.keys())
                    if not self.oww_n_models:
                        logger.error("No wake word models loaded.")
                    for model_key in self.owwModel.models.keys():
                        logger.info(
                            "Successfully loaded openwakeword model: "
                            f"{model_key}"
                        )
                except Exception as e:
                    logger.exception(
                        "Error initializing openwakeword "
                        f"wake word detection engine: {e}"
                    )
                    raise
                logger.debug("Open wake word detection engine initialized successfully")
            else:
                logger.exception(f"Wakeword engine {self.wakeword_backend} unknown/unsupported or wake_words not specified. Please specify one of: pvporcupine, openwakeword.")

        # Setup voice activity detection model WebRTC (8-space indent for this block)
        try:
            # 12-space indent for lines inside try/except
            logger.info("Initializing WebRTC voice with "
                         f"Sensitivity {webrtc_sensitivity}"
                         )
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)
        except Exception as e:
            logger.exception("Error initializing WebRTC voice "
                              f"activity detection engine: {e}"
                              )
            raise # Re-raise exception if WebRTC VAD fails, it's likely critical

        logger.debug("WebRTC VAD voice activity detection " # Back to 8-space indent
                      "engine initialized successfully"
                      )

        # Setup voice activity detection model Silero VAD (8-space indent for this block)
        self.silero_vad_model = None 
        self.silero_vad_utils = None 
        try:
            # 12-space indent for lines inside try/except
            logger.info("Initializing Silero VAD voice activity detection engine..."
                        f"ONNX: {self.silero_use_onnx}")
            self.silero_vad_model, self.silero_vad_utils = torch.hub.load(
                repo_or_dir='/app/silero_assets', 
                model='silero_vad',              
                source='local',
                trust_remote_code=True,          
                onnx=self.silero_use_onnx        
            )
            logger.debug("Silero VAD voice activity detection "
                         "engine initialized successfully"
                         )
        except Exception as e:
            logger.exception(f"Error initializing Silero VAD "
                             f"voice activity detection engine: {e}"
                             )
            # Silero VAD is somewhat optional; log error but don't necessarily raise
            # self.silero_vad_model and self.silero_vad_utils remain None
        
        # Back to 8-space indent for remaining assignments
        self.audio_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       self.pre_recording_buffer_duration)
        )
        self.last_words_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       0.3)
        )
        self.frames = []
        self.last_frames = []

        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
        
        logger.debug('Waiting for main transcription model to start')
        self.main_transcription_ready_event.wait()
        logger.debug('Main transcription model ready')

        self.stdout_thread = threading.Thread(target=self._read_stdout)
        self.stdout_thread.daemon = True
        self.stdout_thread.start()

        logger.debug('RealtimeSTT initialization completed successfully')

    def _start_thread(self, target=None, args=()):
        # Indentation: 4 spaces for method body
        if (platform.system() == 'Linux'):
            # 8 spaces for lines inside if
            thread = threading.Thread(target=target, args=args)
            thread.deamon = True # Note: 'daemon' is the correct spelling
            thread.start()
            return thread
        else:
            # 8 spaces for lines inside else
            thread = mp.Process(target=target, args=args)
            thread.start()
            return thread

    def _read_stdout(self):
        # Indentation: 4 spaces for method body
        while not self.shutdown_event.is_set():
            # 8 spaces for loop body
            try:
                # 12 spaces for try body
                if self.parent_stdout_pipe.poll(0.1):
                    # 16 spaces for if body
                    logger.debug("Receive from stdout pipe")
                    message = self.parent_stdout_pipe.recv()
                    logger.info(message)
            except (BrokenPipeError, EOFError, OSError):
                pass
            except KeyboardInterrupt: 
                logger.info("KeyboardInterrupt in read from stdout detected, exiting...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in read from stdout: {e}", exc_info=True)
                logger.error(traceback.format_exc()) 
                break
        
            time.sleep(0.1)

    def _transcription_worker(*args, **kwargs):
        # Indentation: 4 spaces for method body
        worker = TranscriptionWorker(*args, **kwargs)
        worker.run()

    def _run_callback(self, cb, *args, **kwargs):
        # Indentation: 4 spaces for method body
        if self.start_callback_in_new_thread:
            # 8 spaces for if body
            threading.Thread(target=cb, args=args, kwargs=kwargs, daemon=True).start()
        else:
            # 8 spaces for else body
            cb(*args, **kwargs)

    @staticmethod
    def _audio_data_worker(
        audio_queue,
        target_sample_rate,
        buffer_size,
        input_device_index,
        shutdown_event,
        interrupt_stop_event,
        use_microphone
    ):
        """ Docstring as provided by user ... """
        import pyaudio
        import numpy as np
        # from scipy import signal # Already imported at top level

        if __name__ == '__main__':
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

        def get_highest_sample_rate(audio_interface, device_index):
            """Get the highest supported sample rate for the specified device."""
            try:
                device_info = audio_interface.get_device_info_by_index(device_index)
                logger.debug(f"Retrieving highest sample rate for device index {device_index}: {device_info}")
                max_rate = int(device_info['defaultSampleRate'])
                
                if 'supportedSampleRates' in device_info:
                    supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
                    if supported_rates:
                        max_rate = max(supported_rates)
                
                logger.debug(f"Highest supported sample rate for device index {device_index} is {max_rate}")
                return max_rate
            except Exception as e:
                logger.warning(f"Failed to get highest sample rate: {e}")
                return 48000 

        def initialize_audio_stream(audio_interface, sample_rate, chunk_size):
            nonlocal input_device_index

            def validate_device(device_index):
                """Validate that the device exists and is actually available for input."""
                try:
                    device_info = audio_interface.get_device_info_by_index(device_index)
                    logger.debug(f"Validating device index {device_index} with info: {device_info}")
                    if not device_info.get('maxInputChannels', 0) > 0:
                        logger.debug("Device has no input channels, invalid for recording.")
                        return False
                    
                    test_stream = audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=target_sample_rate, # Use target_sample_rate for test
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=device_index,
                        start=False 
                    )
                    
                    test_stream.start_stream()
                    test_data = test_stream.read(chunk_size, exception_on_overflow=False)
                    test_stream.stop_stream()
                    test_stream.close()
                    
                    if len(test_data) == 0:
                        logger.debug("Device produced no data, invalid for recording.")
                        return False
                    
                    logger.debug(f"Device index {device_index} successfully validated.")
                    return True
                
                except Exception as e:
                    logger.debug(f"Device validation failed for index {device_index}: {e}")
                    return False

            """Initialize the audio stream with error handling."""
            while not shutdown_event.is_set():
                try:
                    input_devices = []
                    device_count = audio_interface.get_device_count()
                    logger.debug(f"Found {device_count} total audio devices on the system.")
                    for i in range(device_count):
                        try:
                            device_info = audio_interface.get_device_info_by_index(i)
                            if device_info.get('maxInputChannels', 0) > 0:
                                input_devices.append(i)
                        except Exception as e:
                            logger.debug(f"Could not retrieve info for device index {i}: {e}")
                            continue
                    
                    logger.debug(f"Available input devices with input channels: {input_devices}")
                    if not input_devices:
                        raise Exception("No input devices found")

                    if input_device_index is None or input_device_index not in input_devices:
                        try:
                            default_device = audio_interface.get_default_input_device_info()
                            logger.debug(f"Default device info: {default_device}")
                            if validate_device(default_device['index']):
                                input_device_index = default_device['index']
                                logger.debug(f"Default device {input_device_index} selected.")
                        except Exception:
                            logger.debug("Default device validation failed, checking other devices...")
                            for device_idx_check in input_devices: # Renamed inner loop variable
                                if validate_device(device_idx_check):
                                    input_device_index = device_idx_check
                                    logger.debug(f"Device {input_device_index} selected.")
                                    break
                            else:
                                raise Exception("No working input devices found")
                    
                    if not validate_device(input_device_index):
                        raise Exception(f"Selected device validation failed for index {input_device_index}")

                    logger.debug(f"Opening stream with device index {input_device_index}, "
                                f"sample_rate={sample_rate}, chunk_size={chunk_size}")
                    stream = audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=input_device_index,
                    )
                    
                    logger.info(f"Microphone connected and validated (device index: {input_device_index}, "
                                f"sample rate: {sample_rate}, chunk size: {chunk_size})")
                    return stream
                
                except Exception as e:
                    logger.error(f"Microphone connection failed: {e}. Retrying...", exc_info=True)
                    input_device_index = None 
                    time.sleep(3) 
                    continue
        
        def preprocess_audio(chunk_data, original_rate, target_rate): # Renamed parameters for clarity
            """Preprocess audio chunk similar to feed_audio method."""
            # chunk_data is expected to be 'bytes' from stream.read or numpy array for feed_audio
            if not isinstance(chunk_data, np.ndarray):
                 # If chunk is bytes, convert to numpy array
                chunk_np = np.frombuffer(chunk_data, dtype=np.int16)
            else:
                chunk_np = chunk_data # Already a numpy array
            
            # Handle stereo to mono conversion if necessary
            if chunk_np.ndim == 2:
                chunk_np = np.mean(chunk_np, axis=1)

            # Resample to target_rate if necessary
            if original_rate != target_rate:
                logger.debug(f"Resampling from {original_rate} Hz to {target_rate} Hz.")
                num_samples = int(len(chunk_np) * target_rate / original_rate)
                chunk_np = signal.resample(chunk_np, num_samples)
            
            return chunk_np.astype(np.int16).tobytes() # Always return bytes Output

        audio_interface = None
        stream = None
        device_sample_rate = None # This will be set during setup_audio
        # Use a common chunk size for reading from PyAudio, can be different from VAD's buffer_size
        pyaudio_read_chunk_size = 1024 

        def setup_audio():
            nonlocal audio_interface, stream, device_sample_rate, input_device_index
            try:
                if audio_interface is None:
                    logger.debug("Creating PyAudio interface...")
                    audio_interface = pyaudio.PyAudio()
                
                if input_device_index is None:
                    try:
                        default_device = audio_interface.get_default_input_device_info()
                        input_device_index = default_device['index']
                        logger.debug(f"No device index supplied; using default device {input_device_index}")
                    except OSError as e:
                        logger.debug(f"Default device retrieval failed: {e}")
                        input_device_index = None # Ensure it's None if retrieval fails
                
                # We'll try 16000 Hz first, then the highest rate we detect, then fallback if needed
                sample_rates_to_try = [16000] # Prefer 16kHz if available
                detected_highest_rate = 48000 # Default fallback highest rate
                if input_device_index is not None:
                    try:
                        detected_highest_rate = get_highest_sample_rate(audio_interface, input_device_index)
                        if detected_highest_rate not in sample_rates_to_try:
                             sample_rates_to_try.append(detected_highest_rate)
                    except Exception as e: # Handle cases where get_highest_sample_rate might fail
                        logger.warning(f"Could not detect highest sample rate for device {input_device_index}: {e}. Using fallback.")
                else: # If no input_device_index, still add a common high rate
                    if 48000 not in sample_rates_to_try:
                        sample_rates_to_try.append(48000)


                logger.debug(f"Sample rates to try for device {input_device_index}: {sample_rates_to_try}")

                for rate_to_try in sample_rates_to_try:
                    try:
                        logger.debug(f"Attempting to initialize audio stream at {rate_to_try} Hz with chunk_size {pyaudio_read_chunk_size}.")
                        # Initialize stream uses pyaudio_read_chunk_size, not the main VAD buffer_size
                        current_stream = initialize_audio_stream(audio_interface, rate_to_try, pyaudio_read_chunk_size)
                        if current_stream is not None:
                            device_sample_rate = rate_to_try # Store the actual rate used
                            stream = current_stream # Assign to the non-local stream
                            logger.debug(
                                f"Audio recording initialized successfully at {device_sample_rate} Hz, "
                                f"reading {pyaudio_read_chunk_size} frames at a time"
                            )
                            return True
                    except Exception as e:
                        logger.warning(f"Failed to initialize audio stream at {rate_to_try} Hz: {e}")
                        continue

                raise Exception("Failed to initialize audio stream with any of the tried sample rates.")
            
            except Exception as e:
                logger.exception(f"Error initializing pyaudio audio recording: {e}")
                if audio_interface:
                    audio_interface.terminate()
                return False

        logger.debug(f"Starting audio data worker with target_sample_rate={target_sample_rate}, "
                    f"VAD buffer_size={buffer_size}, input_device_index={input_device_index}")

        if not setup_audio(): # This sets up audio_interface, stream, and device_sample_rate
            # If setup fails, an exception should have been logged.
            # We might want to signal the main process or retry, but for now, let's exit the worker.
            logger.error("Audio data worker: Failed to set up audio. Shutting down worker.")
            return # Exit the worker thread/process if audio setup fails

        internal_buffer = bytearray()
        # target_chunk_size_bytes is the size of data Silero VAD expects (e.g., 512 samples * 2 bytes/sample = 1024 bytes)
        target_chunk_size_bytes = 2 * buffer_size # buffer_size is Silero's expected sample count

        time_since_last_buffer_message = 0

        try:
            while not shutdown_event.is_set():
                try:
                    # Read from PyAudio stream using its own optimal chunk size
                    raw_mic_data = stream.read(pyaudio_read_chunk_size, exception_on_overflow=False)

                    if use_microphone.value:
                        # Preprocess (includes resampling to target_sample_rate)
                        processed_data_bytes = preprocess_audio(raw_mic_data, device_sample_rate, target_sample_rate)
                        internal_buffer += processed_data_bytes

                        # check if the internal_buffer has reached or exceeded the target_chunk_size_bytes for VAD
                        while len(internal_buffer) >= target_chunk_size_bytes:
                            # Extract one chunk of the target size for VAD
                            chunk_for_vad = internal_buffer[:target_chunk_size_bytes]
                            internal_buffer = internal_buffer[target_chunk_size_bytes:]

                            if time_since_last_buffer_message == 0 or time.time() - time_since_last_buffer_message > 1 : # Log less frequently
                                logger.debug("_audio_data_worker putting VAD-sized audio data into queue.")
                                time_since_last_buffer_message = time.time()
                            
                            audio_queue.put(chunk_for_vad) # Put the VAD-ready chunk
                
                except OSError as e:
                    if hasattr(pyaudio, 'paInputOverflowed') and e.errno == pyaudio.paInputOverflowed: # Check if paInputOverflowed exists
                        logger.warning("Input overflowed. Frame dropped.")
                    else:
                        logger.error(f"OSError during recording: {e}", exc_info=True)
                        logger.error("Attempting to reinitialize the audio stream...")
                        
                        try:
                            if stream:
                                stream.stop_stream()
                                stream.close()
                        except Exception as close_exc:
                            logger.error(f"Error closing stream during reinitialization: {close_exc}")
                        
                        time.sleep(1)
                        if not setup_audio(): # re-initializes stream and device_sample_rate
                            logger.error("Failed to reinitialize audio stream. Exiting audio worker.")
                            break # Exit while loop, and thus the worker
                        else:
                            logger.info("Audio stream reinitialized successfully.") # Renamed from error to info
                    continue # Continue the outer while loop
                
                except Exception as e:
                    logger.error(f"Unknown error during recording: {e}", exc_info=True) # Added exc_info
                    logger.error("Attempting to reinitialize the audio stream...")
                    try:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                    except Exception as close_exc:
                        logger.error(f"Error closing stream during reinitialization: {close_exc}")
                    
                    time.sleep(1)
                    if not setup_audio(): # re-initializes stream and device_sample_rate
                        logger.error("Failed to reinitialize audio stream. Exiting audio worker.")
                        break # Exit while loop, and thus the worker
                    else:
                        logger.info("Audio stream reinitialized successfully.") # Renamed from error to info
                    continue # Continue the outer while loop
        except KeyboardInterrupt:
            interrupt_stop_event.set()
            logger.debug("Audio data worker process finished due to KeyboardInterrupt")
        finally:
            # After recording stops, feed any remaining audio data from internal_buffer
            # This needs to be chunked correctly if not empty
            while len(internal_buffer) >= target_chunk_size_bytes:
                chunk_for_vad = internal_buffer[:target_chunk_size_bytes]
                internal_buffer = internal_buffer[target_chunk_size_bytes:]
                audio_queue.put(chunk_for_vad)
            if internal_buffer: # If there's a remainder less than a full chunk
                logger.debug(f"Putting remaining {len(internal_buffer)} bytes from internal_buffer to queue.")
                # Depending on how the VAD handles partial final chunks, you might pad or send as is.
                # For Silero, it might be better to pad to full size or discard if too small.
                # For now, sending as is, assuming downstream can handle it or _recording_worker pads.
                audio_queue.put(bytes(internal_buffer))


            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            except Exception as e:
                logger.error(f"Error closing stream in finally block: {e}")
            if audio_interface:
                audio_interface.terminate()
            logger.debug("Audio data worker cleanup complete.")


    def wakeup(self):
        self.listen_start = time.time()

    def abort(self):
        state = self.state
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self.interrupt_stop_event.set()
        if self.state != "inactive": 
            self.was_interrupted.wait()
            self._set_state("transcribing")
        self.was_interrupted.clear()
        if self.is_recording: 
            self.stop()
    
    def wait_audio(self):
        try:
            logger.info("Setting listen time")
            if self.listen_start == 0:
                self.listen_start = time.time()

            if not self.is_recording and not self.frames:
                self._set_state("listening")
                self.start_recording_on_voice_activity = True

                logger.debug('Waiting for recording start')
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout=0.02):
                        break

            if self.is_recording:
                self.stop_recording_on_voice_deactivity = True

                logger.debug('Waiting for recording stop')
                while not self.interrupt_stop_event.is_set():
                    if (self.stop_recording_event.wait(timeout=0.02)):
                        break
            
            frames_copy = self.frames[:] # Work with a copy
            if not frames_copy and self.last_frames: # If current frames empty, use last_frames
                frames_copy = self.last_frames[:]


            if not frames_copy:
                logger.warning("wait_audio: No frames available to process.")
                self.audio = np.array([], dtype=np.float32)
                self.frames.clear()
                self.last_frames.clear()
                self.listen_start = 0
                self._set_state("inactive")
                return


            full_audio_array = np.frombuffer(b''.join(frames_copy), dtype=np.int16)
            full_audio = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

            samples_to_keep_for_resume = int(self.sample_rate * self.backdate_resume_seconds)
            new_frames_for_buffer = []

            if samples_to_keep_for_resume > 0:
                samples_to_keep_for_resume = min(samples_to_keep_for_resume, len(full_audio))
                audio_for_resume_buffer = full_audio[-samples_to_keep_for_resume:]
                
                frame_bytes_for_resume = (audio_for_resume_buffer * INT16_MAX_ABS_VALUE).astype(np.int16).tobytes()
                
                # Assuming Silero chunk size for repopulating self.frames for next potential recording session
                vad_chunk_size_bytes = 2 * self.buffer_size 
                for i in range(0, len(frame_bytes_for_resume), vad_chunk_size_bytes):
                    frame_chunk = frame_bytes_for_resume[i:i + vad_chunk_size_bytes]
                    if frame_chunk: 
                        new_frames_for_buffer.append(frame_chunk)
            
            samples_to_remove_from_end = int(self.sample_rate * self.backdate_stop_seconds)
            if samples_to_remove_from_end > 0:
                if samples_to_remove_from_end < len(full_audio):
                    self.audio = full_audio[:-samples_to_remove_from_end]
                    logger.debug(f"Removed {samples_to_remove_from_end} samples "
                        f"({samples_to_remove_from_end/self.sample_rate:.3f}s) from end of audio")
                else:
                    self.audio = np.array([], dtype=np.float32)
                    logger.debug("Cleared audio (samples_to_remove_from_end >= audio length)")
            else:
                self.audio = full_audio
                logger.debug(f"No samples removed, final audio length: {len(self.audio)}")

            self.frames.clear()
            self.last_frames.clear() # Clear last_frames as current recording is processed
            self.frames.extend(new_frames_for_buffer) # Repopulate with resume buffer

            self.backdate_stop_seconds = 0.0
            self.backdate_resume_seconds = 0.0
            self.listen_start = 0
            self._set_state("inactive")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in wait_audio, shutting down")
            self.shutdown()
            raise

    def perform_final_transcription(self, audio_data=None, use_prompt=True): # Renamed audio_bytes to audio_data
        start_time = 0
        with self.transcription_lock:
            if audio_data is None: # Check audio_data
                audio_data = copy.deepcopy(self.audio) # Use self.audio (which is float32 numpy array)

            if audio_data is None or len(audio_data) == 0: # Check audio_data
                # logger.info block was commented out, uncomment if desired
                # logger.info("No audio data available for transcription")
                print("No audio data available for transcription")
                return ""

            try:
                if self.transcribe_count == 0: # This logic seems okay
                    logger.debug("Adding transcription request, no early transcription started")
                    start_time = time.time() 
                    # Ensure audio_data is float32 numpy array as expected by worker
                    if not isinstance(audio_data, np.ndarray) or audio_data.dtype != np.float32:
                        logger.warning("Audio data for transcription is not in expected float32 format. Attempting conversion.")
                        if isinstance(audio_data, bytes):
                            audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
                        elif isinstance(audio_data, np.ndarray) and audio_data.dtype == np.int16:
                            audio_data = audio_data.astype(np.float32) / INT16_MAX_ABS_VALUE
                        else:
                            logger.error("Cannot convert audio_data to required format for transcription.")
                            return "" # Or raise error

                    self.parent_transcription_pipe.send((audio_data, self.language, use_prompt))
                    self.transcribe_count += 1

                while self.transcribe_count > 0:
                    logger.debug(F"Receive from parent_transcription_pipe after sending transcription request, transcribe_count: {self.transcribe_count}")
                    if not self.parent_transcription_pipe.poll(0.1): 
                        if self.interrupt_stop_event.is_set(): 
                            self.was_interrupted.set()
                            self._set_state("inactive")
                            return "" 
                        continue
                    status, result = self.parent_transcription_pipe.recv()
                    self.transcribe_count -= 1

                self.allowed_to_early_transcribe = True
                self._set_state("inactive")
                if status == 'success':
                    segments, info = result # segments is the transcription, info is transcription info
                    self.detected_language = info.language if info.language_probability > 0 else None # Check probability
                    self.detected_language_probability = info.language_probability
                    
                    # Storing the audio_data (float32 numpy array) that was transcribed
                    self.last_transcription_bytes = copy.deepcopy(audio_data) # Store the float32 array
                    # For b64, first convert to int16 bytes if needed for consistency, or directly encode float32 if preferred
                    audio_int16_bytes = (audio_data * INT16_MAX_ABS_VALUE).astype(np.int16).tobytes()
                    self.last_transcription_bytes_b64 = base64.b64encode(audio_int16_bytes).decode('utf-8')
                    
                    transcription_text = self._preprocess_output(segments) # Pass segments (transcription_text)
                    end_time = time.time() 
                    transcription_time = end_time - start_time

                    if start_time: # Only log if start_time was set (i.e., not an early transcription return path)
                        if self.print_transcription_time:
                            print(f"Model {self.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                        else:
                            logger.debug(f"Model {self.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                    return "" if self.interrupt_stop_event.is_set() else transcription_text
                else:
                    logger.error(f"Transcription error: {result}")
                    raise Exception(result) # Re-raise exception 
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}", exc_info=True)
                self._set_state("inactive") # Ensure state is reset on error
                self.transcribe_count = 0 # Reset count on error
                self.allowed_to_early_transcribe = True
                raise # Re-raise exception

    def transcribe(self):
        """Docstring as provided by user..."""
        # self.audio is a float32 numpy array, no need to copy if perform_final_transcription handles it
        audio_to_transcribe = copy.deepcopy(self.audio) # Make a copy to avoid modification during concurrent ops
        self._set_state("transcribing")
        
        callback_result = None
        if self.on_transcription_start:
            # Assuming on_transcription_start might return a value to abort
            callback_result = self.on_transcription_start(audio_to_transcribe) 
        
        # If on_transcription_start returns a value that evaluates to True, it might mean abort.
        # Original code had "if not abort_value:", so if callback_result is None or False, proceed.
        if not callback_result:
            return self.perform_final_transcription(audio_to_transcribe) # Pass the copied audio
        
        # If callback aborted (e.g., returned True), return None or an empty string.
        logger.info("Transcription aborted by on_transcription_start callback.")
        self._set_state("inactive") # Ensure state is reset if aborted here
        return "" # Or None, depending on desired behavior for aborted transcription

    def _process_wakeword(self, data):
        """Docstring as provided by user..."""
        # data is expected to be bytes from audio_queue
        if self.wakeword_backend in {'pvp', 'pvporcupine'}:
            try: # Added try-except for robustness
                pcm = struct.unpack_from(
                    "h" * (len(data) // 2), # Use actual length of data
                    data
                )
                # Ensure self.porcupine is initialized
                if hasattr(self, 'porcupine') and self.porcupine:
                    porcupine_index = self.porcupine.process(pcm)
                    if self.debug_mode:
                        logger.info(f"wake words porcupine_index: {porcupine_index}")
                    return porcupine_index
                else:
                    logger.warning("Porcupine not initialized, cannot process wake word.")
                    return -1
            except struct.error as e:
                logger.error(f"Struct error in _process_wakeword (pvporcupine): {e}. Data length: {len(data)}")
                return -1
            except Exception as e:
                logger.error(f"General error in _process_wakeword (pvporcupine): {e}", exc_info=True)
                return -1


        elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            try: # Added try-except
                # Ensure self.owwModel is initialized
                if not hasattr(self, 'owwModel') or not self.owwModel:
                    logger.warning("OpenWakeWord model not initialized, cannot process wake word.")
                    return -1

                pcm = np.frombuffer(data, dtype=np.int16)
                prediction = self.owwModel.predict(pcm) # This updates self.owwModel.prediction_buffer
                
                max_score = -1.0 # Use float for scores
                max_index = -1
                
                # Check if prediction_buffer has keys and if sensitivities are set
                if not hasattr(self, 'wake_words_sensitivities') or not self.wake_words_sensitivities:
                    # Fallback to a single sensitivity if list is not properly initialized
                    # This part might need adjustment based on how sensitivities are meant to map to oww models
                    current_sensitivity = self.wake_words_sensitivity if hasattr(self, 'wake_words_sensitivity') else 0.5
                else:
                    # This assumes a correspondence by index if multiple models/sensitivities
                    # For OWW, sensitivities might be applied differently or per model.
                    # The original code had 'self.wake_words_sensitivities' but owwModel.predict
                    # internally uses thresholds from the model config or a global one.
                    # For simplicity, using self.wake_words_sensitivity (the float attribute) here.
                    current_sensitivity = self.wake_words_sensitivity


                if self.owwModel.prediction_buffer: # Check if buffer is not empty
                    for idx, mdl_key in enumerate(self.owwModel.prediction_buffer.keys()):
                        scores = list(self.owwModel.prediction_buffer[mdl_key])
                        if scores and scores[-1] >= current_sensitivity and scores[-1] > max_score:
                            max_score = scores[-1]
                            # OWW returns model name if detected, not index directly.
                            # We need to map this back to an "index" if other parts of the code expect it.
                            # For now, returning 0 or a fixed positive value if any model triggers.
                            # Or, if a specific keyword mapping is needed:
                            # for keyword_idx, keyword_model_name in enumerate(self.wake_words_list_for_oww_perhaps):
                            # if mdl_key relates to keyword_model_name: max_index = keyword_idx; break
                            max_index = 0 # Placeholder: treat any oww detection as index 0
                    if self.debug_mode:
                        logger.info(f"wake words oww max_index (derived), max_score: {max_index} {max_score}")
                    return max_index if max_score >= current_sensitivity else -1 # Return based on max_score found
                else:
                    if self.debug_mode:
                        logger.info(f"wake words oww_index: -1 (empty prediction buffer or no scores above sensitivity)")
                    return -1
            except Exception as e:
                logger.error(f"Error processing wake word with OpenWakeWord: {e}", exc_info=True)
                return -1


        if self.debug_mode: # This will only be reached if backend is not matched
            logger.info("wake words no match (unknown backend or error)")
        return -1


    def text(self,
             on_transcription_finished=None,
             ):
        """Docstring as provided by user..."""
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        try:
            self.wait_audio()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in text() method, shutting down.")
            self.shutdown()
            raise 

        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            logger.info("text() returning early due to shutdown or interrupt.")
            return ""

        transcription_result = self.transcribe() # Get transcription result directly

        if on_transcription_finished:
            # Run in a new thread only if callback is provided
            threading.Thread(target=on_transcription_finished,
                            args=(transcription_result,)).start()
            return None # Typically, if async callback, method might return None or a future
        else:
            return transcription_result # Return result directly if no callback


    def format_number(self, num):
        num_str = f"{num:.10f}" 
        integer_part, decimal_part = num_str.split('.')
        result = f"{integer_part[-2:]}.{decimal_part[:2]}"
        return result

    def start(self, frames = None):
        """Docstring as provided by user..."""
        if (time.time() - self.recording_stop_time
                < self.min_gap_between_recordings):
            logger.info("Attempted to start recording "
                         "too soon after stopping."
                         )
            return self

        logger.info("recording started")
        self._set_state("recording")
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        
        self.frames.clear() # Clear existing frames before starting
        if frames: # If initial frames are provided
            # Frames should be a list of byte chunks (VAD-sized)
            self.frames.extend(frames) 
        
        self.is_recording = True
        self.recording_start_time = time.time()
        self.is_silero_speech_active = False # Reset VAD states
        self.is_webrtc_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            self._run_callback(self.on_recording_start)
        return self

    def stop(self,
             backdate_stop_seconds: float = 0.0,
             backdate_resume_seconds: float = 0.0,
        ):
        """Docstring as provided by user..."""
        if self.recording_start_time == 0 : # If not recording, don't stop
            logger.info("Attempted to stop recording when not started.")
            return self

        if (time.time() - self.recording_start_time
                < self.min_length_of_recording):
            logger.info("Attempted to stop recording "
                         "too soon after starting."
                         )
            return self

        logger.info("recording stopped")
        # self.last_frames = copy.deepcopy(self.frames) # Store current frames before clearing for next session
        # The `wait_audio` logic might handle `last_frames` differently, let's defer to it.
        # Here, we just set flags.
        
        self.backdate_stop_seconds = backdate_stop_seconds
        self.backdate_resume_seconds = backdate_resume_seconds
        
        was_recording = self.is_recording # Store current state
        self.is_recording = False # Primary flag to stop recording logic
        
        # VAD states reset
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0 # Reset silero check timer
        
        self.start_recording_event.clear() # Ensure start event is cleared
        self.stop_recording_event.set()    # Signal stop event for wait_audio

        if was_recording: # Only update times and call callback if it was actually recording
            self.recording_stop_time = time.time()
            self.last_recording_start_time = self.recording_start_time
            self.last_recording_stop_time = self.recording_stop_time
            self.recording_start_time = 0 # Reset for next session

            if self.on_recording_stop:
                self._run_callback(self.on_recording_stop)
        return self

    def listen(self):
        """Docstring as provided by user..."""
        logger.info("Entering listen state.") # Added log
        self.listen_start = time.time()
        self._set_state("listening")
        self.start_recording_on_voice_activity = True
        self.clear_audio_queue() # Clear queue when starting to listen

    def feed_audio(self, chunk, original_sample_rate=16000):
        """Docstring as provided by user..."""
        if not hasattr(self, 'feed_buffer'): # Use a separate buffer for feed_audio
            self.feed_buffer = bytearray()

        processed_chunk_bytes = ""
        if isinstance(chunk, np.ndarray):
            # Assuming chunk is float32 if numpy array from external source, normalize and convert
            if chunk.dtype == np.float32:
                chunk_int16 = (chunk * INT16_MAX_ABS_VALUE).astype(np.int16)
            elif chunk.dtype == np.int16:
                chunk_int16 = chunk
            else:
                logger.error(f"feed_audio: Unsupported numpy array dtype {chunk.dtype}")
                return

            # Resample if necessary (using the main class sample_rate as target for VAD)
            if original_sample_rate != self.sample_rate: # self.sample_rate is VAD's target (e.g. 16000)
                num_samples = int(len(chunk_int16) * self.sample_rate / original_sample_rate)
                chunk_resampled = signal.resample(chunk_int16, num_samples)
                processed_chunk_bytes = chunk_resampled.astype(np.int16).tobytes()
            else:
                processed_chunk_bytes = chunk_int16.tobytes()
        
        elif isinstance(chunk, bytes):
            # Assuming bytes are int16, resample if necessary
            if original_sample_rate != self.sample_rate:
                chunk_np = np.frombuffer(chunk, dtype=np.int16)
                num_samples = int(len(chunk_np) * self.sample_rate / original_sample_rate)
                chunk_resampled = signal.resample(chunk_np, num_samples)
                processed_chunk_bytes = chunk_resampled.astype(np.int16).tobytes()
            else:
                processed_chunk_bytes = chunk # Already bytes and correct sample rate
        else:
            logger.error(f"feed_audio: Unsupported chunk type {type(chunk)}")
            return

        self.feed_buffer += processed_chunk_bytes
        
        # Use self.buffer_size (VAD's expected sample count) to determine chunk size in bytes
        vad_chunk_size_bytes = 2 * self.buffer_size 

        while len(self.feed_buffer) >= vad_chunk_size_bytes:
            to_process = self.feed_buffer[:vad_chunk_size_bytes]
            self.feed_buffer = self.feed_buffer[vad_chunk_size_bytes:]
            self.audio_queue.put(to_process)


    def set_microphone(self, microphone_on=True):
        logger.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on

    def shutdown(self):
        """Docstring as provided by user..."""
        with self.shutdown_lock:
            if self.is_shut_down:
                return

            print("\033[91mRealtimeSTT shutting down\033[0m")
            self.is_shut_down = True # Set this early
            
            self.shutdown_event.set() # Signal all processes/threads
            self.interrupt_stop_event.set() # Also signal interrupts

            # Force events for methods like wait_audio to unblock
            self.start_recording_event.set()
            self.stop_recording_event.set()
            
            self.is_recording = False # Ensure recording stops
            self.is_running = False   # Stop main loops in workers

            logger.debug('Finishing recording thread (_recording_worker)')
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2) # Reduced timeout
                if self.recording_thread.is_alive():
                    logger.warning("Recording thread did not join in time.")


            logger.debug('Terminating reader process (_audio_data_worker)')
            if hasattr(self, 'reader_process') and self.reader_process and self.reader_process.is_alive():
                self.reader_process.join(timeout=5) # Allow more time for PyAudio cleanup
                if self.reader_process.is_alive():
                    logger.warning("Reader process did not terminate in time. Terminating forcefully.")
                    self.reader_process.terminate() # Force terminate if stuck

            logger.debug('Terminating transcription process (_transcription_worker)')
            if hasattr(self, 'transcript_process') and self.transcript_process and self.transcript_process.is_alive():
                # Ensure parent_transcription_pipe is sent a sentinel or closed to unblock worker
                try:
                    if self.parent_transcription_pipe:
                         # self.parent_transcription_pipe.send(None) # Optional: send sentinel
                         self.parent_transcription_pipe.close()
                except Exception as e:
                    logger.warning(f"Error closing parent_transcription_pipe: {e}")

                self.transcript_process.join(timeout=5)
                if self.transcript_process.is_alive():
                    logger.warning("Transcript process did not terminate in time. Terminating forcefully.")
                    self.transcript_process.terminate()
            
            # Close pipes if not already closed by the processes
            try:
                if self.parent_transcription_pipe: self.parent_transcription_pipe.close()
                # child_transcription_pipe is managed by the worker
                if self.parent_stdout_pipe: self.parent_stdout_pipe.close()
                # child_stdout_pipe is managed by the worker
            except Exception as e:
                logger.warning(f"Error closing pipes during shutdown: {e}")


            logger.debug('Finishing realtime thread (_realtime_worker)')
            if self.realtime_thread and self.realtime_thread.is_alive():
                self.realtime_thread.join(timeout=2)
                if self.realtime_thread.is_alive():
                    logger.warning("Realtime thread did not join in time.")


            if self.enable_realtime_transcription:
                if hasattr(self, 'realtime_model_type') and self.realtime_model_type:
                    del self.realtime_model_type
                    self.realtime_model_type = None
            
            # Porcupine cleanup if used
            if hasattr(self, 'porcupine') and self.porcupine:
                try:
                    self.porcupine.delete()
                    self.porcupine = None
                    logger.debug("Porcupine instance deleted.")
                except Exception as e:
                    logger.error(f"Error deleting Porcupine instance: {e}")

            # OpenWakeWord model cleanup (if it has explicit cleanup)
            if hasattr(self, 'owwModel') and self.owwModel:
                # owwModel might not have an explicit delete, Python's GC handles it
                del self.owwModel 
                self.owwModel = None
                logger.debug("OpenWakeWord model deleted (or reference removed).")

            gc.collect()
            logger.info("RealtimeSTT shutdown complete.")


    def _recording_worker(self):
        """Docstring as provided by user..."""
        if self.use_extended_logging:
            logger.debug('Debug: Entering _recording_worker try block')

        last_inner_try_time = 0
        try:
            if self.use_extended_logging:
                logger.debug('Debug: Initializing _recording_worker variables')
            time_since_last_buffer_message = 0
            was_recording_state = False # More descriptive name
            delay_was_passed = False
            # wakeword_detected_time = None # This is self.wake_word_detect_time
            wakeword_samples_to_remove = 0 # Initialize to 0
            self.allowed_to_early_transcribe = True

            if self.use_extended_logging:
                logger.debug('Debug: Starting _recording_worker main loop')
            
            while self.is_running:
                if self.use_extended_logging and last_inner_try_time: # Reduce logging frequency
                    processing_time_check = time.time() - last_inner_try_time
                    if processing_time_check > 0.1:
                        logger.warning(f'### WARNING: _recording_worker loop took {processing_time_check:.3f}s')
                last_inner_try_time = time.time()
                
                try:
                    try:
                        # data is expected to be a VAD-sized chunk of bytes (e.g., 1024 bytes for 512 samples int16)
                        data = self.audio_queue.get(timeout=0.02) # Increased timeout slightly
                        self.last_words_buffer.append(data) # This buffer is for what? context?
                    except queue.Empty:
                        if not self.is_running: break
                        time.sleep(TIME_SLEEP / 2) # Shorter sleep if queue empty but still running
                        continue
                    except BrokenPipeError: # This might happen if a process writing to queue dies
                        logger.error("BrokenPipeError in _recording_worker audio_queue.get()", exc_info=True)
                        self.is_running = False # Stop this worker
                        break


                    if self.on_recorded_chunk:
                        self._run_callback(self.on_recorded_chunk, data)

                    if self.handle_buffer_overflow and self.audio_queue.qsize() > self.allowed_latency_limit:
                        logger.warning(f"Audio queue size ({self.audio_queue.qsize()}) exceeds "
                                         f"latency limit ({self.allowed_latency_limit}). Discarding oldest chunks.")
                        while self.audio_queue.qsize() > self.allowed_latency_limit:
                            try:
                                self.audio_queue.get_nowait() # Discard
                            except queue.Empty:
                                break #Queue empty, stop discarding
                
                except Exception as e_outer_loop: # Catch unexpected errors in getting/handling data
                    logger.error(f"Error in _recording_worker data handling part: {e_outer_loop}", exc_info=True)
                    time.sleep(0.1) # Brief pause before retrying loop
                    continue


                failed_stop_attempt = False # Reset per iteration

                if not self.is_recording: # Processing when NOT in 'recording' state
                    time_since_listen_start = (time.time() - self.listen_start) if self.listen_start else float('inf')

                    # Wake word activation delay logic
                    current_delay_passed_state = (self.wake_word_activation_delay == 0 or \
                                                 (self.listen_start and time_since_listen_start > self.wake_word_activation_delay))

                    if current_delay_passed_state and not delay_was_passed:
                        if self.use_wake_words and self.wake_word_activation_delay > 0 and self.on_wakeword_timeout:
                            logger.debug('Debug: Calling on_wakeword_timeout due to delay passed.')
                            self._run_callback(self.on_wakeword_timeout)
                    delay_was_passed = current_delay_passed_state
                    
                    # State setting based on conditions
                    if not self.recording_stop_time: # i.e. not recently stopped
                        if self.use_wake_words and delay_was_passed and not self.wakeword_detected:
                            self._set_state("wakeword")
                        elif self.listen_start: # Actively listening (VAD or post-wakeword)
                            self._set_state("listening")
                        else: # Truly inactive
                            self._set_state("inactive")
                    
                    # Wake word detection logic
                    if self.use_wake_words and delay_was_passed and not self.wakeword_detected:
                        wakeword_index = self._process_wakeword(data)
                        if wakeword_index >= 0:
                            logger.info(f"Wake word detected (index {wakeword_index}). Transitioning to listen for speech.")
                            self.wake_word_detect_time = time.time()
                            wakeword_samples_to_remove = int(self.sample_rate * self.wake_word_buffer_duration * 2) # bytes
                            self.wakeword_detected = True
                            self.listen_start = time.time() # Start "listening" phase for VAD
                            self._set_state("listening") # Update state
                            if self.on_wakeword_detected:
                                self._run_callback(self.on_wakeword_detected)
                            self.clear_audio_queue() # Clear queue after wake word to avoid processing it as speech

                    # VAD-based recording start logic
                    # Conditions to check for VAD:
                    # 1. start_recording_on_voice_activity is True (set by listen() or text())
                    # 2. OR wakeword was detected (self.wakeword_detected is True)
                    # AND ( (NOT using wake words) OR (wake word activation delay passed OR wake word was detected) )

                    should_check_vad = self.start_recording_on_voice_activity or self.wakeword_detected
                    
                    if should_check_vad:
                        self._check_voice_activity(data) # This updates self.is_silero_speech_active and self.is_webrtc_speech_active
                        if self._is_voice_active(): # Checks combination of WebRTC and Silero based on flags
                            logger.info("Voice activity detected by VAD. Starting recording.")
                            if self.on_vad_start: self._run_callback(self.on_vad_start)
                            
                            # Collect current audio_buffer content for pre-roll
                            pre_roll_frames = list(self.audio_buffer)
                            self.audio_buffer.clear()
                            
                            self.start(frames=pre_roll_frames) # Start recording, pass pre-roll.
                            
                            # wakeword_samples_to_remove is in bytes. self.frames contains byte chunks.
                            if self.wakeword_detected and wakeword_samples_to_remove > 0:
                                temp_frames_bytes = b''.join(self.frames)
                                if wakeword_samples_to_remove < len(temp_frames_bytes):
                                    remaining_bytes = temp_frames_bytes[wakeword_samples_to_remove:]
                                    self.frames.clear()
                                    vad_chunk_sz = 2 * self.buffer_size
                                    for i in range(0, len(remaining_bytes), vad_chunk_sz):
                                        self.frames.append(remaining_bytes[i:i+vad_chunk_sz])
                                    logger.debug(f"Removed {wakeword_samples_to_remove} bytes for wake word.")
                                else: # wake word removal would empty frames
                                    self.frames.clear()
                                    logger.debug("Wake word removal resulted in empty frames.")
                                wakeword_samples_to_remove = 0 # Reset

                            self.start_recording_on_voice_activity = False # VAD now handled by recording state
                            if self.silero_vad_model: self.silero_vad_model.reset_states()
                        # else: (No voice active yet, keep buffering data in audio_buffer)
                        #    pass 
                    
                    if self.speech_end_silence_start != 0: # If previously measuring silence but not recording
                        self.speech_end_silence_start = 0
                        if self.on_turn_detection_stop:
                            self._run_callback(self.on_turn_detection_stop)
                
                else: # Processing when self.is_recording is True
                    self.frames.append(data) # Add current data chunk to actual recording frames

                    if self.stop_recording_on_voice_deactivity:
                        # Determine speech using the configured method
                        is_speech_now = self._is_silero_speech(data) if (self.silero_deactivity_detection and self.silero_vad_model) \
                                        else self._is_webrtc_speech(data, all_frames_must_be_true=True) # WebRTC needs all sub-frames silent

                        if not is_speech_now: # Silence detected
                            if self.speech_end_silence_start == 0 and \
                               (time.time() - self.recording_start_time > self.min_length_of_recording): # Ensure min recording length
                                self.speech_end_silence_start = time.time()
                                self.awaiting_speech_end = True # Flag for realtime worker
                                if self.on_turn_detection_start:
                                    self._run_callback(self.on_turn_detection_start)
                            
                            # Early transcription logic
                            if self.speech_end_silence_start and self.early_transcription_on_silence > 0 and \
                               (time.time() - self.speech_end_silence_start > self.early_transcription_on_silence / 1000.0) and \
                               self.allowed_to_early_transcribe and self.frames: # ensure early_transcription_on_silence is in seconds
                                
                                logger.debug("Attempting early transcription due to silence.")
                                audio_array_early = np.frombuffer(b''.join(self.frames), dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
                                
                                with self.transcription_lock: # Protect transcribe_count and pipe send
                                    self.parent_transcription_pipe.send((audio_array_early, self.language, True)) # True for use_prompt
                                    self.transcribe_count += 1
                                self.allowed_to_early_transcribe = False # Prevent rapid early transcriptions

                        else: # Speech continues
                            self.awaiting_speech_end = False
                            if self.speech_end_silence_start != 0: # Was silent, now speech again
                                logger.info("Speech resumed, resetting silence timer.")
                                self.speech_end_silence_start = 0
                                if self.on_turn_detection_stop:
                                    self._run_callback(self.on_turn_detection_stop)
                                self.allowed_to_early_transcribe = True # Allow early transcription again

                        # Check for end of recording due to prolonged silence
                        if self.speech_end_silence_start and \
                           (time.time() - self.speech_end_silence_start >= self.post_speech_silence_duration):
                            
                            logger.info(f"Prolonged silence ({self.post_speech_silence_duration}s) detected. Stopping recording.")
                            if self.on_vad_stop: self._run_callback(self.on_vad_stop)
                            
                            # self.frames.append(data) # Already added at the start of 'else self.is_recording'
                            self.stop() # This sets is_recording = False, and sets stop_recording_event
                            
                            if not self.is_recording: # if stop() was successful
                                if self.speech_end_silence_start != 0:
                                    self.speech_end_silence_start = 0 # Reset here too
                                    if self.on_turn_detection_stop:
                                        self._run_callback(self.on_turn_detection_stop)
                            else: # stop() failed (e.g., min recording length not met by stop())
                                failed_stop_attempt = True 
                            self.awaiting_speech_end = False


                # Common logic for end of loop iteration
                if not self.is_recording and was_recording_state: # Just stopped recording
                    self.stop_recording_on_voice_deactivity = False # Reset for next session
                    self.audio_buffer.clear() # Clear pre-roll buffer after recording stops

                if time.time() - self.silero_check_time > 0.1 : # Reset silero check time if too old (original logic)
                    self.silero_check_time = 0 

                # Handle wake word timeout (if wake word was detected but no speech followed)
                if self.wakeword_detected and self.wake_word_detect_time and \
                   (time.time() - self.wake_word_detect_time > self.wake_word_timeout) and \
                   not self.is_recording : # Only timeout if not already recording
                    
                    logger.info("Wake word timeout: No speech detected after wake word.")
                    self.wake_word_detect_time = 0
                    self.wakeword_detected = False # Reset wake word state
                    self.listen_start = 0 # Stop "active listening" phase for VAD
                    if self.on_wakeword_timeout:
                        self._run_callback(self.on_wakeword_timeout)
                    self._set_state("wakeword") # Or inactive, depending on desired flow

                was_recording_state = self.is_recording # Update for next iteration

                # Buffer data if not recording or if measuring end-of-speech silence
                if not self.is_recording or self.speech_end_silence_start != 0:
                     # Ensure 'data' is defined; it might not be if queue was empty and loop continued
                    if 'data' in locals() and data:
                        self.audio_buffer.append(data)


        except Exception as e:
            if not self.interrupt_stop_event.is_set() and not self.shutdown_event.is_set(): # Log only if not intentional stop
                logger.error(f"Unhandled exception in _recording_worker: {e}", exc_info=True)
            # No re-raise, allow thread to exit gracefully if possible or rely on shutdown
        finally:
            logger.debug('Exiting _recording_worker method')


    def _realtime_worker(self):
        """Docstring as provided by user..."""
        try:
            logger.debug('Starting realtime worker')
            if not self.enable_realtime_transcription:
                logger.debug('Realtime transcription not enabled. Exiting worker.')
                return

            last_transcription_time = time.time()

            while self.is_running:
                if not self.is_recording: # Only transcribe if actively recording
                    time.sleep(self.realtime_processing_pause / 2 if self.realtime_processing_pause > 0.02 else 0.01) # Sleep if not recording
                    continue

                # Wait for processing pause or state change
                sleep_start_time = time.time()
                while (time.time() - last_transcription_time) < self.realtime_processing_pause:
                    if not self.is_running or not self.is_recording or self.enable_realtime_transcription == False: # Check enable_realtime_transcription dynamically
                        logger.debug("Realtime worker: condition change, breaking sleep.")
                        break 
                    # Shorter sleep for responsiveness
                    time.sleep(0.005) 
                
                if not self.is_running or not self.is_recording or self.enable_realtime_transcription == False: continue # Recheck conditions after sleep

                if self.awaiting_speech_end: # If VAD is waiting for silence to confirm end of speech, pause RT
                    time.sleep(0.01)
                    continue
                
                if not self.frames: # No frames to transcribe
                    last_transcription_time = time.time() # Reset timer to avoid immediate re-transcription
                    continue

                last_transcription_time = time.time() # Update after processing pause

                # Make a safe copy of frames for transcription
                current_frames_for_rt = list(self.frames) # Shallow copy of list
                if not current_frames_for_rt: continue

                try:
                    # Convert current frames to a NumPy array for transcription
                    audio_array_rt = np.frombuffer(b''.join(current_frames_for_rt), dtype=np.int16)
                    if audio_array_rt.size == 0: # Skip if no audio data
                        continue 
                    audio_array_rt = audio_array_rt.astype(np.float32) / INT16_MAX_ABS_VALUE
                except Exception as e_conv:
                    logger.error(f"Error converting frames for realtime transcription: {e_conv}", exc_info=True)
                    continue


                logger.debug(f"Realtime worker: processing {len(audio_array_rt)} samples.")
                
                realtime_segments = None # To hold segments from transcription
                info_rt = None # To hold info object

                if self.use_main_model_for_realtime:
                    with self.transcription_lock: # Ensure thread-safe access if main model is shared
                        try:
                            # Send to main transcription worker, which expects float32 numpy array
                            self.parent_transcription_pipe.send((audio_array_rt, self.language, True)) # True for use_prompt
                            # This recv needs a timeout or non-blocking check if main worker is busy
                            if self.parent_transcription_pipe.poll(timeout=self.realtime_processing_pause * 0.8): # Shorter timeout
                                status, result = self.parent_transcription_pipe.recv()
                                if status == 'success':
                                    realtime_segments_raw, info_rt = result # result is (transcription_text, info_object)
                                    # The main model returns joined text, not segments. We need to adapt.
                                    # For now, let's assume it can be wrapped or used as is.
                                    # This needs careful thought on how 'segments' are handled downstream.
                                    # If _preprocess_output expects segments (list of seg objects), this won't work directly.
                                    # Assuming result[0] is the transcribed text string.
                                    # To make it work with downstream `_preprocess_output(realtime_segments, ...)` which expects segments,
                                    # we might need to wrap this string into a mock segment object or list.
                                    # For now, assign the string, and _preprocess_output might need adjustment or this path needs rework.
                                    # Let's assume `result[0]` (the text) is what we need for `realtime_text_from_model`.
                                    realtime_text_from_model = realtime_segments_raw #This is a string
                                    logger.debug(f"Realtime text (main model): {realtime_text_from_model}")
                                else:
                                    logger.error(f"Realtime transcription error (main model): {result}")
                                    continue
                            else:
                                logger.warning("Realtime transcription (main model) timed out or no data.")
                                # Decrement count if we added it, though main model path might not increment global self.transcribe_count here
                                # This part needs careful review if main model is used for RT.
                                continue
                        except Exception as e:
                            logger.error(f"Error in realtime transcription (main model pipeline): {e}", exc_info=True)
                            continue
                else: # Use dedicated realtime model
                    if not hasattr(self, 'realtime_model_type') or not self.realtime_model_type:
                        logger.warning("Realtime model not available, skipping RT transcription.")
                        continue
                    
                    if self.normalize_audio:
                        if audio_array_rt.size > 0:
                            peak = np.max(np.abs(audio_array_rt))
                            if peak > 0: audio_array_rt = (audio_array_rt / peak) * 0.95
                    
                    try:
                        # faster_whisper transcribe returns (segments, info)
                        # segments is an iterator of Segment objects
                        rt_segments_iter, info_rt = self.realtime_model_type.transcribe(
                            audio_array_rt,
                            language=self.language if self.language else None,
                            beam_size=self.beam_size_realtime,
                            initial_prompt=self.initial_prompt_realtime,
                            suppress_tokens=self.suppress_tokens,
                            # batch_size for BatchedInferencePipeline, not for base WhisperModel
                            **( {"batch_size": self.realtime_batch_size} if isinstance(self.realtime_model_type, BatchedInferencePipeline) else {} ),
                            vad_filter=self.faster_whisper_vad_filter # This VAD is part of faster_whisper
                        )
                        # Convert iterator to list to reuse segments
                        realtime_segments = list(rt_segments_iter) 
                        realtime_text_from_model = " ".join(seg.text for seg in realtime_segments).strip()
                        logger.debug(f"Realtime text (dedicated model): {realtime_text_from_model}")

                    except Exception as e_rt_transcribe:
                        logger.error(f"Error during dedicated realtime model transcription: {e_rt_transcribe}", exc_info=True)
                        continue # Skip this iteration

                # Process transcription result
                if info_rt: # Ensure info_rt is not None
                    self.detected_realtime_language = info_rt.language if info_rt.language_probability > 0 else None
                    self.detected_realtime_language_probability = info_rt.language_probability
                
                # If using main model, realtime_text_from_model is already a string.
                # If using dedicated, it's joined from segments.
                # The _preprocess_output expects the raw text (or segments if it's adapted)
                
                # Check recording state again, as it might change during transcription
                if self.is_recording and (time.time() - self.recording_start_time > self.init_realtime_after_seconds):
                    # Current text from model
                    current_transcribed_text = realtime_text_from_model # This is a string now

                    self.text_storage.append(current_transcribed_text)
                    if len(self.text_storage) > 10: # Limit storage size
                        self.text_storage.pop(0)

                    # Stabilized text logic
                    if len(self.text_storage) >= 2:
                        common_prefix = os.path.commonprefix(self.text_storage[-2:]) # Compare last two
                        if len(common_prefix) >= len(self.realtime_stabilized_safetext):
                            self.realtime_stabilized_safetext = common_prefix
                    
                    # Determine text for stabilized callback
                    # If current transcription has diverged significantly from stabilized
                    match_pos = self._find_tail_match_in_text(self.realtime_stabilized_safetext, current_transcribed_text)
                    
                    text_for_stabilized_cb = ""
                    if match_pos >= 0: # Found a match
                        text_for_stabilized_cb = self.realtime_stabilized_safetext + current_transcribed_text[match_pos:]
                    else: # No good match, restart stabilized text from current if it's long enough, or use old one
                        text_for_stabilized_cb = self.realtime_stabilized_safetext if len(self.realtime_stabilized_safetext) > len(current_transcribed_text) else current_transcribed_text
                        if len(current_transcribed_text) > 3: # Heuristic: consider current text as new base for stabilization
                             self.realtime_stabilized_safetext = current_transcribed_text


                    # Processed text for callbacks (ensure_sentence_starting_uppercase=True, ensure_sentence_ends_with_period=True for preview)
                    processed_current_text = self._preprocess_output(current_transcribed_text, preview=True)
                    processed_stabilized_text = self._preprocess_output(text_for_stabilized_cb, preview=True)

                    if self.on_realtime_transcription_stabilized:
                        self._run_callback(self._on_realtime_transcription_stabilized, processed_stabilized_text)
                    if self.on_realtime_transcription_update:
                        self._run_callback(self._on_realtime_transcription_update, processed_current_text)

        except Exception as e:
            if not self.is_shut_down: # Log only if not part of normal shutdown
                logger.error(f"Unhandled exception in _realtime_worker: {e}", exc_info=True)
        finally:
            logger.debug('Exiting _realtime_worker method')


    def _is_silero_speech(self, chunk_bytes): # Renamed 'chunk' to 'chunk_bytes' for clarity
        """Docstring as provided by user..."""
        if not self.silero_vad_model: 
            # logger.warning("Silero VAD model not loaded. Cannot detect speech via Silero.") # Can be noisy
            return False # Or True if you want to bypass Silero check when model not loaded

        # chunk_bytes is expected to be raw bytes, already at SAMPLE_RATE (16000) due to _audio_data_worker's preprocessing
        # So, no resampling should be needed here if _audio_data_worker guarantees it.
        # If sample_rate of AudioToTextRecorder can be other than 16000, then _audio_data_worker
        # must output at SAMPLE_RATE for this function.
        # Assuming chunk_bytes is already 16kHz from audio_queue.

        self.silero_working = True # Flag to indicate processing
        try:
            audio_chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16)
            # Ensure conversion to float32 and normalization
            audio_float32 = audio_chunk_np.astype(np.float32) / INT16_MAX_ABS_VALUE 
            
            # Silero model expects a PyTorch tensor
            audio_tensor = torch.from_numpy(audio_float32)
            
            vad_prob = self.silero_vad_model(audio_tensor, self.sample_rate).item() # Use self.sample_rate
            
            # Sensitivity: silero_sensitivity of 0.5 means threshold of 0.5 for speech.
            # Higher sensitivity (e.g., 0.8) means lower threshold (1-0.8 = 0.2), easier to detect speech.
            # Original code: vad_prob > (1 - self.silero_sensitivity)
            # If silero_sensitivity = 0.4 (default), threshold is 0.6. speech if vad_prob > 0.6
            # If silero_sensitivity = 0.8, threshold is 0.2. speech if vad_prob > 0.2
            # This seems counter-intuitive. Usually higher sensitivity value means MORE sensitive.
            # Let's assume `self.silero_sensitivity` IS the threshold directly (0 to 1).
            # Speech if vad_prob > self.silero_sensitivity
            # threshold = self.silero_sensitivity # Direct interpretation
            
            # Reverting to original interpretation for now:
            threshold = 1.0 - self.silero_sensitivity # Higher sensitivity value = lower probability threshold for speech.

            current_silero_speech_active = vad_prob > threshold

            if current_silero_speech_active and not self.is_silero_speech_active and self.use_extended_logging:
                logger.info(f"{bcolors.OKGREEN}Silero VAD detected speech (prob: {vad_prob:.2f} > thresh: {threshold:.2f}){bcolors.ENDC}")
            elif not current_silero_speech_active and self.is_silero_speech_active and self.use_extended_logging:
                logger.info(f"{bcolors.WARNING}Silero VAD detected silence (prob: {vad_prob:.2f} <= thresh: {threshold:.2f}){bcolors.ENDC}")
            
            self.is_silero_speech_active = current_silero_speech_active
        except Exception as e:
            logger.error(f"Error in _is_silero_speech: {e}", exc_info=True)
            self.is_silero_speech_active = False # Default to no speech on error
        finally:
            self.silero_working = False
        return self.is_silero_speech_active


    def _is_webrtc_speech(self, chunk_bytes, all_frames_must_be_true=False): # Renamed
        """Docstring as provided by user..."""
        # Expects chunk_bytes to be 16kHz from audio_queue
        speech_str = f"{bcolors.OKGREEN}WebRTC VAD detected speech{bcolors.ENDC}"
        silence_str = f"{bcolors.WARNING}WebRTC VAD detected silence{bcolors.ENDC}"

        # WebRTC VAD expects 10, 20, or 30 ms frames.
        # Sample rate must be 8000, 16000, 32000, or 48000 Hz.
        # We assume self.sample_rate is 16000 as per constants.
        
        # For 16000 Hz, 10ms = 160 samples, 20ms = 320 samples, 30ms = 480 samples.
        # Bytes per sample = 2. So, 320 bytes, 640 bytes, 960 bytes.
        
        frame_duration_ms = 20 # Use 20ms frames for WebRTC
        samples_per_frame = int(self.sample_rate * (frame_duration_ms / 1000.0))
        bytes_per_frame = samples_per_frame * 2 # 2 bytes for int16

        if len(chunk_bytes) % bytes_per_frame != 0:
            # This can happen if VAD buffer_size is not a multiple of WebRTC frame size.
            # e.g. Silero uses 512 samples (1024 bytes). WebRTC 20ms = 320 samples (640 bytes).
            # logger.warning(f"WebRTC: chunk_bytes length {len(chunk_bytes)} is not a multiple of bytes_per_frame {bytes_per_frame}. Truncating.")
            # We can only process full frames.
            pass # Process as many full frames as possible.
            
        num_frames_in_chunk = len(chunk_bytes) // bytes_per_frame
        if num_frames_in_chunk == 0:
            # logger.debug("WebRTC: Chunk too small for any full frame.")
            # If chunk is smaller than a frame, it's probably silence or not enough data yet.
            # To be safe, if we can't determine, assume previous state or silence.
            # self.is_webrtc_speech_active remains unchanged or set to False.
            return self.is_webrtc_speech_active # Or False if preferred default

        speech_frames_count = 0
        try:
            for i in range(num_frames_in_chunk):
                start = i * bytes_per_frame
                end = start + bytes_per_frame
                frame = chunk_bytes[start:end]
                if self.webrtc_vad_model.is_speech(frame, self.sample_rate):
                    speech_frames_count += 1
                    if not all_frames_must_be_true: # If any frame has speech, overall chunk has speech
                        if self.debug_mode: logger.info(f"WebRTC: Speech in frame {i+1}/{num_frames_in_chunk}")
                        if not self.is_webrtc_speech_active and self.use_extended_logging: logger.info(speech_str)
                        self.is_webrtc_speech_active = True
                        return True 
            
            # If loop completes and all_frames_must_be_true is False, it means no speech frame was found
            if not all_frames_must_be_true:
                if self.debug_mode: logger.info(f"WebRTC: No speech in any of {num_frames_in_chunk} frames.")
                if self.is_webrtc_speech_active and self.use_extended_logging: logger.info(silence_str)
                self.is_webrtc_speech_active = False
                return False

            # Logic for all_frames_must_be_true = True
            current_webrtc_speech_active = (speech_frames_count == num_frames_in_chunk)
            if self.debug_mode:
                logger.info(f"WebRTC (all_frames_must_be_true): {speech_frames_count}/{num_frames_in_chunk} speech frames. Detected: {current_webrtc_speech_active}")

            if current_webrtc_speech_active and not self.is_webrtc_speech_active and self.use_extended_logging: logger.info(speech_str)
            elif not current_webrtc_speech_active and self.is_webrtc_speech_active and self.use_extended_logging: logger.info(silence_str)
            
            self.is_webrtc_speech_active = current_webrtc_speech_active
            return current_webrtc_speech_active

        except Exception as e:
            logger.error(f"Error in _is_webrtc_speech: {e}", exc_info=True)
            self.is_webrtc_speech_active = False # Default to False on error
            return False


    def _check_voice_activity(self, data):
        """Docstring as provided by user..."""
        # This method is called when not recording, to see if VAD triggers a recording.
        # It updates self.is_webrtc_speech_active and, if WebRTC is active,
        # potentially self.is_silero_speech_active (in a thread).
        
        self._is_webrtc_speech(data, all_frames_must_be_true=False) # Fast check

        if self.is_webrtc_speech_active:
            # If WebRTC detects speech, and Silero model is available and not already working
            if self.silero_vad_model and not self.silero_working:
                self.silero_working = True # Prevent re-entry
                # Run Silero check in a thread to not block the main recording loop
                threading.Thread(
                    target=self._is_silero_speech,
                    args=(data,), daemon=True).start() # Ensure thread is daemon
            # If Silero is not used/loaded, is_webrtc_speech_active alone might be used by _is_voice_active logic.
        # else: WebRTC found no speech, self.is_silero_speech_active will not be updated by this call.
        # _is_voice_active() will then evaluate based on current states.


    def clear_audio_queue(self):
        """Docstring as provided by user..."""
        logger.debug("Clearing audio_buffer and audio_queue.") # Added log
        self.audio_buffer.clear()
        try:
            while True: # Empty the multiprocessing queue
                self.audio_queue.get_nowait()
        except queue.Empty: # Expected exception
            pass
        except Exception as e: # Catch other potential errors like BrokenPipeError if process died
            logger.warning(f"Exception while clearing audio_queue: {e}")


    def _is_voice_active(self):
        """Docstring as provided by user..."""
        # This method determines if speech is considered active to START a recording.
        # Behavior might depend on whether Silero is used for primary detection or just deactivity.
        
        # If silero_deactivity_detection is True, Silero is mainly for *stopping*.
        # For *starting*, WebRTC might be primary, or a combination.
        # The original logic in _recording_worker for starting:
        # `if self._is_voice_active(): self.start()`
        # So this function's return value directly controls start of recording after VAD check.
        
        # Let's assume: if Silero model is loaded, it's preferred. Otherwise, fall back to WebRTC.
        if self.silero_vad_model:
            # If Silero is primary for starting, then rely on its state.
            # Note: _is_silero_speech could be running in a thread via _check_voice_activity,
            # so self.is_silero_speech_active might not be instantly updated.
            # This could lead to a slight delay if WebRTC is faster.
            # For now, using the current states:
            return self.is_webrtc_speech_active and self.is_silero_speech_active
        else: # Silero not loaded, rely only on WebRTC
            return self.is_webrtc_speech_active


    def _set_state(self, new_state):
        """Docstring as provided by user..."""
        if new_state == self.state:
            return

        old_state = self.state
        self.state = new_state
        logger.info(f"State changed from '{old_state}' to '{new_state}'")

        callbacks_to_run = []
        if old_state == "listening" and self.on_vad_detect_stop:
            callbacks_to_run.append(self.on_vad_detect_stop)
        elif old_state == "wakeword" and self.on_wakeword_detection_end:
            callbacks_to_run.append(self.on_wakeword_detection_end)
        
        spinner_text = None
        spinner_interval = None

        if new_state == "listening":
            if self.on_vad_detect_start: callbacks_to_run.append(self.on_vad_detect_start)
            spinner_text = "speak now"
            spinner_interval = 250
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start: callbacks_to_run.append(self.on_wakeword_detection_start)
            spinner_text = f"say {self.wake_words}" if self.wake_words else "listening for wake word"
            spinner_interval = 500
        elif new_state == "transcribing":
            spinner_text = "transcribing"
            spinner_interval = 50
        elif new_state == "recording":
            spinner_text = "recording"
            spinner_interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None # Ensure spinner is recreated next time

        for cb in callbacks_to_run: # Run all collected callbacks
            self._run_callback(cb)

        if spinner_text:
            self._set_spinner(spinner_text)
            if self.spinner and self.halo and spinner_interval:
                self.halo._interval = spinner_interval


    def _set_spinner(self, text):
        """Docstring as provided by user..."""
        if self.spinner:
            if self.halo is None:
                try: # Halo can sometimes fail on certain terminals/environments
                    self.halo = halo.Halo(text=text, spinner='dots') # Default spinner
                    self.halo.start()
                except Exception as e:
                    logger.warning(f"Failed to create Halo spinner: {e}. Disabling spinner.")
                    self.spinner = False # Disable spinner if it fails
                    self.halo = None
            else:
                self.halo.text = text


    def _preprocess_output(self, text_or_segments, preview=False):
        """Docstring as provided by user..."""
        # This function needs to handle both a string and a list/iterator of segments
        if isinstance(text_or_segments, str):
            current_text = text_or_segments
        else: # Assuming it's an iterable of Segment objects
            current_text = " ".join(seg.text for seg in text_or_segments).strip()
        
        processed_text = re.sub(r'\s+', ' ', current_text.strip())

        if self.ensure_sentence_starting_uppercase and processed_text:
            processed_text = processed_text[0].upper() + processed_text[1:]
        
        if not preview and self.ensure_sentence_ends_with_period:
            if processed_text and processed_text[-1].isalnum():
                processed_text += '.'
        
        return processed_text


    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):
        """Docstring as provided by user..."""
        if not text1 or not text2: return -1 # Handle empty strings
        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1

        target_substring = text1[-length_of_match:]
        
        # Search from end of text2 for better performance with long texts
        for i in range(len(text2) - length_of_match, -1, -1): # Iterate backwards
            if text2[i : i + length_of_match] == target_substring:
                return i + length_of_match # Return end position of match in text2
        return -1


    def _on_realtime_transcription_stabilized(self, text):
        """Docstring as provided by user..."""
        if self.on_realtime_transcription_stabilized and self.is_recording : # Check is_recording
            self._run_callback(self.on_realtime_transcription_stabilized, text)


    def _on_realtime_transcription_update(self, text):
        """Docstring as provided by user..."""
        if self.on_realtime_transcription_update and self.is_recording: # Check is_recording
            self._run_callback(self.on_realtime_transcription_update, text)


    def __enter__(self):
        """Docstring as provided by user..."""
        return self


    def __exit__(self, exc_type, exc_value, traceback_data): # Renamed traceback to traceback_data
        """Docstring as provided by user..."""
        self.shutdown()