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
    main_loop = None  # This will hold our primary event loop

    async def send_to_client(message):
        global client_websocket, logger
        if client_websocket:
            try:
                await client_websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected while trying to send a message.")
                client_websocket = None # Clear stale websocket
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
        # Only print to console if not sending to a client or for debugging
        # print(f"\r{text}", flush=True, end='') # Avoid cluttering server log if client connected

    recorder_config = {
        'spinner': False,
        'use_microphone': False, # Important for server usage
        'model': 'large-v2', # Consider changing to a smaller model if large-v2 is too slow or resource-intensive
        'language': 'en',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.7,
        'min_length_of_recording': 0.1, # From 0 to 0.1 to avoid empty recordings
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.05, # From 0 to 0.05
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
                    full_sentence = recorder.text() # This is a blocking call
                    if full_sentence:
                        logger.info(f"Full sentence: {full_sentence}")
                        if main_loop and main_loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                send_to_client(json.dumps({
                                    'type': 'fullSentence',
                                    'text': full_sentence
                                })), main_loop)
                    else:
                        # If recorder.text() returns empty, sleep briefly to yield control
                        # This happens if no speech is detected for a while
                        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.01), main_loop)
                else: # Recorder not initialized
                    asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), main_loop)


        except Exception as e:
            logger.error(f"Error in recorder thread: {e}", exc_info=True)
            is_running = False # Stop if recorder thread crashes
        finally:
            logger.info("Recorder thread finishing.")
            if recorder:
                recorder.stop() # Ensure recorder resources are released

    def decode_and_resample(audio_data, original_sample_rate, target_sample_rate=16000):
        global logger
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            if original_sample_rate == target_sample_rate:
                return audio_data # No resampling needed

            num_original_samples = len(audio_np)
            if num_original_samples == 0:
                return audio_data # Empty audio

            num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)
            if num_target_samples == 0:
                return np.array([], dtype=np.int16).tobytes() # Resampled to empty

            resampled_audio = resample(audio_np, num_target_samples)
            return resampled_audio.astype(np.int16).tobytes()
        except Exception as e:
            logger.error(f"Error in resampling audio from {original_sample_rate} to {target_sample_rate}: {e}")
            return audio_data # Return original data on error

    async def audio_handler(websocket):
        global client_websocket, logger, recorder, recorder_ready, is_running
        logger.info("Client connected")
        client_websocket = websocket

        try:
            if not recorder_ready.wait(timeout=10): # Wait for recorder to be ready
                logger.error("Recorder not ready after 10s, closing connection.")
                await websocket.send(json.dumps({"type": "error", "message": "Server recorder not ready."}))
                await websocket.close()
                return

            async for message in websocket:
                if not is_running: break # Exit if server is shutting down

                if isinstance(message, bytes):
                    try:
                        # Assuming metadata is prefixed as before.
                        # If your client sends raw audio, this part needs to change.
                        metadata_length = int.from_bytes(message[:4], byteorder='little')
                        metadata_json = message[4:4+metadata_length].decode('utf-8')
                        metadata = json.loads(metadata_json)
                        sample_rate = metadata.get('sampleRate', 16000) # Default if not provided
                                                
                        audio_chunk = message[4+metadata_length:] 

                        if audio_chunk:
                            resampled_chunk = decode_and_resample(audio_chunk, sample_rate, 16000)
                            if recorder:
                                recorder.feed_audio(resampled_chunk)
                        else:
                            logger.warning("Received empty audio chunk after metadata.")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Metadata JSONDecodeError: {e}. Raw message (first 100 bytes): {message[:100]}")
                    except ValueError as e:
                        logger.error(f"ValueError processing message (likely metadata issue): {e}")
                    except Exception as e:
                        logger.error(f"Error processing audio message: {e}", exc_info=True)
                elif isinstance(message, str):
                    logger.info(f"Received text message from client: {message}")
                    # Handle potential text commands if you add any, e.g., for stopping
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
                client_websocket = None # Clear the global ref if this was the active client

    async def main_server_logic():
        global main_loop, is_running, logger, recorder_thread
        main_loop = asyncio.get_running_loop()

        recorder_thread = threading.Thread(target=run_recorder, daemon=True)
        recorder_thread.start()

        # Wait for the recorder to be ready before starting the server
        if not recorder_ready.wait(timeout=30): # Increased timeout
            logger.error("RealtimeSTT recorder failed to initialize in 30 seconds. Server not starting.")
            is_running = False
            return

        server_host = "0.0.0.0"
        server_port = 7860
        logger.info(f"Server starting, listening on {server_host}:{server_port}. Press Ctrl+C to stop.")

        stop_event = asyncio.Event() # For graceful shutdown

        try:
            async with websockets.serve(audio_handler, server_host, server_port):
                await stop_event.wait()  # Run forever until stop_event is set
        except OSError as e:
            logger.error(f"OSError starting server (likely port {server_port} already in use): {e}")
            is_running = False # Ensure recorder thread also knows to stop
        except Exception as e:
            logger.error(f"Server encountered an unexpected error: {e}", exc_info=True)
            is_running = False
        finally:
            logger.info("Server shutdown sequence initiated.")
            is_running = False # Signal all loops to stop
            if recorder_thread and recorder_thread.is_alive():
                logger.info("Waiting for recorder thread to join...")
                recorder_thread.join(timeout=5) # Wait for recorder thread to finish
                if recorder_thread.is_alive():
                    logger.warning("Recorder thread did not join in time.")
            if recorder: # Ensure final cleanup of recorder
                recorder.stop()
                recorder.shutdown()
            logger.info("Main server logic finished.")


    try:
        asyncio.run(main_server_logic())
    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received by asyncio.run, application shutting down...")
    finally:
        is_running = False # Final confirmation to stop all processes
        logger.info("Application shutdown complete.")
