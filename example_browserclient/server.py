import sys
import os
import torch # For PyTorch version and CUDA checks

print("Starting server, please wait...")

# --- PyTorch and CUDA Diagnostics ---
logger_for_diag = logging.getLogger("STARTUP_DIAG") # Use a temp logger or print
logger_for_diag.addHandler(logging.StreamHandler(sys.stdout))
logger_for_diag.setLevel(logging.INFO)

logger_for_diag.info(f"Python version: {sys.version}")
logger_for_diag.info(f"PyTorch version: {torch.__version__}")
logger_for_diag.info(f"CUDA available to PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger_for_diag.info(f"PyTorch CUDA version (compiled with): {torch.version.cuda}")
    logger_for_diag.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    try:
        current_device_idx = torch.cuda.current_device()
        logger_for_diag.info(f"Current CUDA device index: {current_device_idx}")
        logger_for_diag.info(f"Device name: {torch.cuda.get_device_name(current_device_idx)}")
    except Exception as e:
        logger_for_diag.error(f"Error getting CUDA device details: {e}")
else:
    logger_for_diag.warning("CUDA is NOT available to PyTorch. Application will likely run on CPU.")
# --- End Diagnostics ---


print(f"Working directory: {os.getcwd()}")
print(f"Attempting import: from RealtimeSTT.RealtimeSTT import AudioToTextRecorder")
try:
    from RealtimeSTT.RealtimeSTT import AudioToTextRecorder
    print("Successfully imported AudioToTextRecorder from RealtimeSTT.RealtimeSTT")
except ImportError as e:
    print(f"CRITICAL: Failed to import AudioToTextRecorder even with direct path: {e}")
    print(f"Current sys.path: {sys.path}")
    # Simplified listing for brevity
    print(f"Contents of /app: {os.listdir('/app') if os.path.exists('/app') else 'N/A'}")
    print(f"Contents of /app/RealtimeSTT/RealtimeSTT: {os.listdir('/app/RealtimeSTT/RealtimeSTT') if os.path.exists('/app/RealtimeSTT/RealtimeSTT') else 'N/A'}")
    sys.exit("Exiting due to critical import error.")


import asyncio
import websockets
import threading
import numpy as np
from scipy.signal import resample
import json
import logging # Regular logger setup after diags

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__) # Main application logger
logging.getLogger('websockets').setLevel(logging.WARNING)
if _onnxruntime_found := True: # Placeholder based on your previous logs, RealtimeSTT checks this
    try:
        import onnxruntime
        logger.info(f"ONNX Runtime version: {onnxruntime.__version__}")
        logger.info(f"ONNX Runtime available providers: {onnxruntime.get_available_providers()}")
    except ImportError:
        _onnxruntime_found = False
        logger.warning("ONNX Runtime library not found.")
    except Exception as e:
        logger.error(f"Error getting ONNX Runtime details: {e}")


is_running = True
recorder = None
recorder_ready = threading.Event()
client_websocket = None
main_loop = None

async def send_to_client(message):
    global client_websocket, logger
    if client_websocket:
        try:
            await client_websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected while trying to send a message.")
            client_websocket = None
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")

def text_detected(text):
    global main_loop, logger
    if main_loop and main_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            send_to_client(json.dumps({
                'type': 'realtime',
                'text': text
            })), main_loop)
    else:
        logger.warning("Main event loop not available for text_detected callback.")

recorder_config = {
    'spinner': False,
    'use_microphone': False,
    'model': 'large-v2', # Main Whisper model
    'language': 'en',
    'device': "cuda",    # For Whisper models (main and realtime)
    'silero_use_onnx': True, # Use ONNX for Silero VAD
    # 'silero_assets_path' is NOT a parameter here. It expects "silero_assets" in CWD.
    'silero_sensitivity': 0.4,
    'webrtc_sensitivity': 2, # webrtc_sensitivity is used if Silero VAD fails or is disabled
    'post_speech_silence_duration': 0.7,
    'min_length_of_recording': 0.1,
    'min_gap_between_recordings': 0.0,
    'enable_realtime_transcription': True,
    'realtime_processing_pause': 0.05,
    'realtime_model_type': 'tiny.en', # Realtime Whisper model
    'on_realtime_transcription_stabilized': text_detected,
}

def run_recorder():
    global recorder, main_loop, is_running, logger, recorder_ready
    try:
        logger.info(f"Initializing RealtimeSTT with config: {json.dumps(recorder_config, default=lambda o: '<object>')}")
        # Check if silero_assets directory is accessible from CWD (/app)
        silero_assets_in_cwd_path = os.path.join(os.getcwd(), "silero_assets")
        if os.path.isdir(silero_assets_in_cwd_path):
            logger.info(f"Confirmed 'silero_assets' directory exists at: {silero_assets_in_cwd_path}")
            logger.info(f"Contents of '{silero_assets_in_cwd_path}': {os.listdir(silero_assets_in_cwd_path)}")
        else:
            logger.warning(f"'silero_assets' directory NOT FOUND at expected location: {silero_assets_in_cwd_path}")
            logger.warning("Silero VAD initialization might fail or use fallback mechanisms.")

        recorder = AudioToTextRecorder(**recorder_config)
        
        if recorder and hasattr(recorder, 'use_silero_vad') and recorder.use_silero_vad:
            logger.info("RealtimeSTT initialized successfully with Silero VAD enabled.")
            if hasattr(recorder, 'silero_vad_model') and recorder.silero_vad_model:
                 logger.info(f"Silero VAD model type: {type(recorder.silero_vad_model)}")
                 if recorder.silero_use_onnx:
                     logger.info(f"Silero VAD ONNX session providers: {recorder.silero_vad_model.sess.get_providers() if hasattr(recorder.silero_vad_model, 'sess') else 'N/A'}")

        elif recorder:
            logger.warning("RealtimeSTT initialized, but Silero VAD appears to be disabled or failed to load.")
        else:
             logger.error("RealtimeSTT recorder object is None after initialization attempt.")
        
        recorder_ready.set() 

        while is_running:
            if recorder:
                full_sentence = recorder.text() 
                if full_sentence:
                    logger.info(f"Full sentence: {full_sentence}")
                    if main_loop and main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            send_to_client(json.dumps({
                                'type': 'fullSentence',
                                'text': full_sentence
                            })), main_loop)
                else:
                    if main_loop and main_loop.is_running():
                           asyncio.run_coroutine_threadsafe(asyncio.sleep(0.01), main_loop)
                    else:
                        threading.Event().wait(0.01)
            else:
                logger.warning("Recorder object not available in run_recorder loop. Possible init failure.")
                if main_loop and main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), main_loop)
                else:
                    threading.Event().wait(0.1)

    except NameError: 
        logger.error("AudioToTextRecorder is not defined. Import failed earlier.")
        is_running = False
        recorder_ready.set() 
    except Exception as e:
        logger.error(f"Error in recorder thread: {e}", exc_info=True)
        is_running = False 
        recorder_ready.set() 
    finally:
        logger.info("Recorder thread finishing.")
        if recorder:
            try:
                recorder.stop()
                logger.info("RealtimeSTT recorder stopped.")
            except Exception as e:
                logger.error(f"Exception during recorder.stop(): {e}", exc_info=True)


def decode_and_resample(audio_data, original_sample_rate, target_sample_rate=16000):
    # This function is not currently used in your server.py as audio comes from client.
    # If it were, ensure numpy and scipy are robustly handled.
    global logger
    try:
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        # ... (rest of function)
        return audio_data # Placeholder if not fully implemented or needed
    except Exception as e:
        logger.error(f"Error in resampling: {e}", exc_info=True)
        return audio_data


async def audio_handler(websocket):
    global client_websocket, logger, recorder, recorder_ready, is_running
    logger.info("Client connected to audio_handler.")
    client_websocket = websocket 

    try:
        if not recorder_ready.wait(timeout=20): 
            logger.error("Recorder not ready event timeout. Closing connection.")
            await websocket.send(json.dumps({"type": "error", "message": "Server recorder initialization timeout."}))
            await websocket.close(); return
        
        if recorder is None:
            logger.error("Recorder object is None. Cannot process audio.")
            await websocket.send(json.dumps({"type": "error", "message": "Server STT engine failed to initialize."}))
            await websocket.close(); return
        
        logger.info("Recorder is ready. Awaiting audio data.")
        await websocket.send(json.dumps({"type": "info", "message": "Server ready. Send audio."}))

        async for message in websocket:
            if not is_running: logger.info("Server shutting down, breaking audio handler loop."); break
            if isinstance(message, bytes):
                if message:
                    if recorder: recorder.feed_audio(message)
                    else: logger.warning("Recorder not available to feed audio.")
                else: logger.warning("Received empty audio chunk.")
            elif isinstance(message, str):
                logger.info(f"Received text message: {message}")
                # Handle text commands if any
    except websockets.exceptions.ConnectionClosedOK: logger.info("Client disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e: logger.warning(f"Client connection closed with error: {e}")
    except Exception as e: logger.error(f"Error in websocket audio_handler: {e}", exc_info=True)
    finally:
        logger.info("Client session ended.")
        if client_websocket == websocket: client_websocket = None


async def main_server_logic():
    global main_loop, is_running, logger, recorder_thread
    main_loop = asyncio.get_running_loop() 

    recorder_thread = threading.Thread(target=run_recorder, daemon=True)
    recorder_thread.start()

    initialization_timeout = 120 # Increased for potentially slow model downloads/loads on first RunPod start
    logger.info(f"Waiting up to {initialization_timeout}s for recorder_ready event...")
    if not recorder_ready.wait(timeout=initialization_timeout):
        logger.error(f"RealtimeSTT recorder failed to signal readiness within {initialization_timeout}s.")
        is_running = False 
        if recorder_thread.is_alive(): recorder_thread.join(timeout=10)
        return 
    
    if recorder is None:
        logger.error("Recorder object is None after recorder_ready. Cannot start WebSocket server.")
        is_running = False
        if recorder_thread.is_alive(): recorder_thread.join(timeout=10)
        return
    
    # Log VAD status after recorder_ready is confirmed and recorder object exists
    if hasattr(recorder, 'use_silero_vad') and recorder.use_silero_vad:
        logger.info(f"Silero VAD is enabled in the recorder. ONNX mode: {recorder.silero_use_onnx}.")
    else:
        logger.warning("Silero VAD is disabled in the recorder, or attribute 'use_silero_vad' not found.")

    server_host = "0.0.0.0"
    server_port = 7860 
    logger.info(f"WebSocket server starting, listening on ws://{server_host}:{server_port}")
    
    stop_event = asyncio.Event() 
    try:
        async with websockets.serve(audio_handler, server_host, server_port, max_size=None, ping_interval=20, ping_timeout=20):
            await stop_event.wait() 
    except OSError as e:
        logger.error(f"OSError starting server on {server_host}:{server_port}: {e}")
        is_running = False
    except Exception as e:
        logger.error(f"Main server logic error: {e}", exc_info=True)
        is_running = False
    finally:
        logger.info("Server shutdown sequence initiated.")
        is_running = False 
        stop_event.set() 
        if 'recorder_thread' in globals() and recorder_thread.is_alive(): 
            logger.info("Waiting for recorder thread to join...")
            recorder_thread.join(timeout=10) 
            if recorder_thread.is_alive(): logger.warning("Recorder thread did not join in time.")
        logger.info("Main server logic finished.")

if __name__ == '__main__':
    if 'AudioToTextRecorder' in globals():
        try: asyncio.run(main_server_logic())
        except KeyboardInterrupt: logger.info("KeyboardInterrupt. Shutting down...")
        except Exception as e: logger.error(f"Unhandled exception in __main__: {e}", exc_info=True)
        finally:
            is_running = False 
            logger.info("Application shutdown complete.")
    else:
        logger.error("AudioToTextRecorder not imported. Application cannot start.")
        logger.info("Application shutdown due to import failure.")
