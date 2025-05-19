import sys
import os

print("Starting server, please wait...")
# sys.path should be correctly set by Dockerfile's ENV PYTHONPATH to include /app
# The RealtimeSTT repo is cloned to /app/RealtimeSTT
# The package itself is at /app/RealtimeSTT/RealtimeSTT
# The working import, as per diagnostics, is:
print(f"Working directory: {os.getcwd()}")
print(f"Attempting import: from RealtimeSTT.RealtimeSTT import AudioToTextRecorder")
try:
    from RealtimeSTT.RealtimeSTT import AudioToTextRecorder
    print("Successfully imported AudioToTextRecorder from RealtimeSTT.RealtimeSTT")
except ImportError as e:
    print(f"CRITICAL: Failed to import AudioToTextRecorder even with direct path: {e}")
    # Add more debug info if this fails, though it shouldn't based on previous log
    print(f"Current sys.path: {sys.path}")
    print(f"Contents of /app: {os.listdir('/app') if os.path.exists('/app') else 'N/A'}")
    print(f"Contents of /app/RealtimeSTT: {os.listdir('/app/RealtimeSTT') if os.path.exists('/app/RealtimeSTT') else 'N/A'}")
    print(f"Contents of /app/RealtimeSTT/RealtimeSTT: {os.listdir('/app/RealtimeSTT/RealtimeSTT') if os.path.exists('/app/RealtimeSTT/RealtimeSTT') else 'N/A'}")
    sys.exit("Exiting due to critical import error.")


import asyncio
import websockets
import threading
import numpy as np
from scipy.signal import resample # Ensure scipy is in your requirements-gpu.txt
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logging.getLogger('websockets').setLevel(logging.WARNING)

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
    'model': 'large-v2',
    'language': 'en',
    'device': "cuda",
    'silero_use_onnx': True,
    'silero_vad_path': '/app/RealtimeSTT/silero_assets', # CORRECTED PARAMETER NAME
    'silero_sensitivity': 0.4,
    'webrtc_sensitivity': 2,
    'post_speech_silence_duration': 0.7,
    'min_length_of_recording': 0.1,
    'min_gap_between_recordings': 0.0,
    'enable_realtime_transcription': True,
    'realtime_processing_pause': 0.05,
    'realtime_model_type': 'tiny.en', 
    'on_realtime_transcription_stabilized': text_detected,
}

def run_recorder():
    global recorder, main_loop, is_running, logger, recorder_ready
    try:
        logger.info(f"Initializing RealtimeSTT with config: {json.dumps(recorder_config, default=lambda o: '<object>')}")
        recorder = AudioToTextRecorder(**recorder_config)
        logger.info("RealtimeSTT initialized successfully.")
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
                logger.warning("Recorder object not available in run_recorder loop.")
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
    global logger
    try:
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        if original_sample_rate == target_sample_rate:
            return audio_data 
        num_original_samples = len(audio_np)
        if num_original_samples == 0:
            return audio_data 
        num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)
        if num_target_samples == 0:
            return np.array([], dtype=np.int16).tobytes() 
        resampled_audio = resample(audio_np, num_target_samples)
        return resampled_audio.astype(np.int16).tobytes()
    except Exception as e:
        logger.error(f"Error in resampling audio from {original_sample_rate} to {target_sample_rate}: {e}", exc_info=True)
        return audio_data 

async def audio_handler(websocket):
    global client_websocket, logger, recorder, recorder_ready, is_running
    logger.info("Client connected to audio_handler.")
    client_websocket = websocket 

    try:
        if not recorder_ready.wait(timeout=20): 
            logger.error("Recorder not ready within timeout. Closing connection.")
            await websocket.send(json.dumps({"type": "error", "message": "Server recorder initialization timeout."}))
            await websocket.close()
            return
        
        logger.info("Recorder is ready. Awaiting audio data.")
        await websocket.send(json.dumps({"type": "info", "message": "Server ready. Send audio."}))

        async for message in websocket:
            if not is_running:
                logger.info("Server shutting down, breaking audio handler loop.")
                break
            if isinstance(message, bytes):
                audio_chunk_from_client = message 
                client_sample_rate = 16000 
                if audio_chunk_from_client:
                    processed_chunk = audio_chunk_from_client 
                    if recorder:
                        recorder.feed_audio(processed_chunk)
                    else:
                        logger.warning("Recorder not available to feed audio.")
                else:
                    logger.warning("Received empty audio chunk from client.")
            
            elif isinstance(message, str):
                logger.info(f"Received text message from client: {message}")
                try:
                    data = json.loads(message)
                    if data.get("type") == "control" and data.get("command") == "stop_processing":
                        logger.info("Client requested to stop processing (example command).")
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from text message: {message}")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Client disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Error in websocket audio_handler: {e}", exc_info=True)
    finally:
        logger.info("Client session ended.")
        if client_websocket == websocket: 
            client_websocket = None

async def main_server_logic():
    global main_loop, is_running, logger, recorder_thread
    main_loop = asyncio.get_running_loop() 

    recorder_thread = threading.Thread(target=run_recorder, daemon=True)
    recorder_thread.start()

    initialization_timeout = 90 
    if not recorder_ready.wait(timeout=initialization_timeout):
        logger.error(f"RealtimeSTT recorder failed to initialize in {initialization_timeout}s. Server not starting.")
        is_running = False 
        if recorder_thread.is_alive():
            recorder_thread.join(timeout=5) 
        if 'AudioToTextRecorder' not in globals() or recorder is None and not recorder_ready.is_set(): 
            logger.error("Exiting main_server_logic as recorder initialization seems to have failed critically.")
            return


    server_host = "0.0.0.0"
    server_port = 7860 
    logger.info(f"WebSocket server starting, listening on ws://{server_host}:{server_port}")
    
    stop_event = asyncio.Event() 

    try:
        async with websockets.serve(
            audio_handler,
            server_host,
            server_port,
            max_size=None, 
            ping_interval=20, 
            ping_timeout=20   
        ):
            await stop_event.wait() 
    except OSError as e:
        logger.error(f"OSError starting server (Port {server_port} likely in use or permission issue): {e}")
        is_running = False
    except Exception as e:
        logger.error(f"Main server logic encountered an unexpected error: {e}", exc_info=True)
        is_running = False
    finally:
        logger.info("Server shutdown sequence initiated.")
        is_running = False 
        stop_event.set() 
        if 'recorder_thread' in globals() and recorder_thread.is_alive(): 
            logger.info("Waiting for recorder thread to join...")
            recorder_thread.join(timeout=10) 
            if recorder_thread.is_alive():
                logger.warning("Recorder thread did not join in time.")
        logger.info("Main server logic finished.")

if __name__ == '__main__':
    if 'AudioToTextRecorder' in globals():
        try:
            asyncio.run(main_server_logic())
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down application...")
        except Exception as e: 
            logger.error(f"Unhandled exception in asyncio.run: {e}", exc_info=True)
        finally:
            is_running = False 
            logger.info("Application shutdown complete.")
    else:
        logger.error("Application cannot start because AudioToTextRecorder could not be imported.")
        logger.info("Application shutdown due to import failure.")
