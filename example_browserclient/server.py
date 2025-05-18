if __name__ == '__main__':
    print("Starting server, please wait...")
    from RealtimeSTT import AudioToTextRecorder
    import asyncio
    import websockets
    import threading
    import numpy as np
    from scipy.signal import resample # Make sure scipy is in your requirements.txt
    import json
    import logging
    import sys

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

    recorder_config = {
        'spinner': False,
        'use_microphone': False,
        'model': 'large-v2',
        'language': 'en',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.7,
        'min_length_of_recording': 0.1,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.05,
        'realtime_model_type': 'tiny.en',
        'on_realtime_transcription_stabilized': text_detected,
    }

    def run_recorder():
        global recorder, main_loop, is_running, logger, recorder_ready
        try:
            logger.info("Initializing RealtimeSTT...")
            recorder = AudioToTextRecorder(**recorder_config)
            logger.info("RealtimeSTT initialized")
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
                    if main_loop and main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), main_loop)
                    else:
                        threading.Event().wait(0.1)
        except Exception as e:
            logger.error(f"Error in recorder thread: {e}", exc_info=True)
            is_running = False
        finally:
            logger.info("Recorder thread finishing.")
            if recorder:
                recorder.stop()

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
            logger.error(f"Error in resampling audio from {original_sample_rate} to {target_sample_rate}: {e}")
            return audio_data

    async def audio_handler(websocket):
        global client_websocket, logger, recorder, recorder_ready, is_running
        logger.info("Client connected")
        client_websocket = websocket

        try:
            if not recorder_ready.wait(timeout=10):
                logger.error("Recorder not ready after 10s, closing connection.")
                await websocket.send(json.dumps({"type": "error", "message": "Server recorder not ready."}))
                await websocket.close()
                return

            async for message in websocket:
                if not is_running: break

                if isinstance(message, bytes):
                    # THIS IS THE PRIMARY PATH FOR AUDIO FROM REALTIME1.HTML
                    try:
                        audio_chunk_from_client = message 
                        client_assumed_sample_rate = 16000

                        if audio_chunk_from_client:
                            processed_chunk = decode_and_resample(
                                audio_chunk_from_client, 
                                client_assumed_sample_rate, 
                                16000 
                            )
                            if recorder:
                                recorder.feed_audio(processed_chunk)
                        else:
                            logger.warning("Received empty audio chunk.")
                    
                    except Exception as e:
                        logger.error(f"Error processing raw audio message: {e}", exc_info=True)
                
                elif isinstance(message, str):
                    logger.info(f"Received text message from client: {message}")
                    # If you need to handle JSON commands from client, parse here
                    # try:
                    #     command_data = json.loads(message)
                    #     # process command_data
                    # except json.JSONDecodeError:
                    #     logger.warning(f"Received non-JSON text message: {message}")
                    pass

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Client disconnected gracefully.")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Client connection closed with error: {e}")
        except Exception as e:
            logger.error(f"Error in websocket handler: {e}", exc_info=True)
        finally:
            logger.info("Client session ended.")
            if client_websocket == websocket:
                client_websocket = None

    async def main_server_logic():
        global main_loop, is_running, logger
        main_loop = asyncio.get_running_loop()
        recorder_thread = threading.Thread(target=run_recorder, daemon=True)
        recorder_thread.start()

        if not recorder_ready.wait(timeout=60):
            logger.error("RealtimeSTT recorder failed to initialize in 60 seconds. Server not starting.")
            is_running = False
            if recorder_thread.is_alive(): recorder_thread.join(timeout=1)
            return

        server_host = "0.0.0.0"
        server_port = 7860
        logger.info(f"Server starting, listening on {server_host}:{server_port}. Press Ctrl+C to stop.")
        stop_event = asyncio.Event()

        try:
            async with websockets.serve(audio_handler, server_host, server_port, max_size=None):
                await stop_event.wait()
        except OSError as e:
            logger.error(f"OSError starting server (likely port {server_port} already in use): {e}")
            is_running = False
        except Exception as e:
            logger.error(f"Server encountered an unexpected error: {e}", exc_info=True)
            is_running = False
        finally:
            logger.info("Server shutdown sequence initiated.")
            is_running = False
            stop_event.set()
            if recorder_thread and recorder_thread.is_alive():
                logger.info("Waiting for recorder thread to join...")
                recorder_thread.join(timeout=5)
                if recorder_thread.is_alive():
                    logger.warning("Recorder thread did not join in time.")
            logger.info("Main server logic finished.")

    try:
        asyncio.run(main_server_logic())
    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received by asyncio.run, application shutting down...")
    finally:
        is_running = False
        logger.info("Application shutdown complete.")
