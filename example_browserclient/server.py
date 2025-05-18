if __name__ == '__main__':
    print("Starting server, please wait...")
    from RealtimeSTT import AudioToTextRecorder # This will now import from the cloned repo
    import asyncio
    import websockets
    import threading
    import numpy as np
    from scipy.signal import resample # Ensure scipy is in your requirements-gpu.txt
    import json
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logging.getLogger('websockets').setLevel(logging.WARNING) # Quieten verbose websockets logs

    is_running = True
    recorder = None
    recorder_ready = threading.Event()
    client_websocket = None
    main_loop = None # To schedule coroutines from the recorder thread

    async def send_to_client(message):
        global client_websocket, logger
        if client_websocket:
            try:
                await client_websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected while trying to send a message.")
                client_websocket = None # Clear the stale websocket
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")

    def text_detected(text):
        """Callback for realtime transcription updates."""
        global main_loop, logger
        if main_loop and main_loop.is_running():
            # Schedule the send_to_client coroutine on the main asyncio event loop
            asyncio.run_coroutine_threadsafe(
                send_to_client(json.dumps({
                    'type': 'realtime',
                    'text': text
                })), main_loop)
        else:
            logger.warning("Main event loop not available for text_detected callback.")


    # Configuration for AudioToTextRecorder
    recorder_config = {
        'spinner': False,
        'use_microphone': False,  # We are feeding audio via websocket
        'model': 'large-v2',      # Using large-v2, can be changed to large-v3 or others
        'language': 'en',
        
        'device': "cuda",         # Use CUDA for main model and VAD (if ONNX)
        'silero_use_onnx': True,  # IMPORTANT: Use ONNX for Silero VAD (hoping the cloned lib handles it correctly now)
        
        'silero_sensitivity': 0.4, # VAD sensitivity
        'webrtc_sensitivity': 2,   # Another VAD option (less relevant if Silero is primary)
        'post_speech_silence_duration': 0.7, # How much silence before finalizing speech
        'min_length_of_recording': 0.1,     # Minimum audio length to process
        'min_gap_between_recordings': 0.0,  # Minimal gap
        'enable_realtime_transcription': True, # Enable intermediate results
        'realtime_processing_pause': 0.05,     # Pause between realtime processing cycles
        'realtime_model_type': 'tiny.en',      # Faster model for realtime, should also run on CUDA
        'on_realtime_transcription_stabilized': text_detected, # Callback for stable realtime text
    }

    def run_recorder():
        """Target function for the recorder thread."""
        global recorder, main_loop, is_running, logger, recorder_ready
        try:
            # Log the config being used, masking sensitive/long data if any
            logger.info(f"Initializing RealtimeSTT with config: {json.dumps(recorder_config, default=lambda o: '<object>')}")
            recorder = AudioToTextRecorder(**recorder_config)
            logger.info("RealtimeSTT initialized successfully.")
            recorder_ready.set() # Signal that recorder is ready

            # Main loop for the recorder thread
            while is_running:
                if recorder:
                    full_sentence = recorder.text() # Blocking call, waits for a full sentence
                    if full_sentence:
                        logger.info(f"Full sentence: {full_sentence}")
                        if main_loop and main_loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                send_to_client(json.dumps({
                                    'type': 'fullSentence',
                                    'text': full_sentence
                                })), main_loop)
                    else:
                        # recorder.text() might return None if shutting down or no speech
                        # Add a small sleep to prevent tight loop if is_running is true but no text
                        if main_loop and main_loop.is_running():
                           asyncio.run_coroutine_threadsafe(asyncio.sleep(0.01), main_loop) # yield to event loop
                        else:
                            threading.Event().wait(0.01) # fallback if no loop
                else:
                    logger.warning("Recorder object not available in run_recorder loop.")
                    if main_loop and main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), main_loop)
                    else:
                        threading.Event().wait(0.1) # Pause if recorder isn't set up

        except Exception as e:
            logger.error(f"Error in recorder thread: {e}", exc_info=True)
            is_running = False # Stop the application on critical recorder error
            recorder_ready.set() # Unblock main thread if it's waiting, even on error
        finally:
            logger.info("Recorder thread finishing.")
            if recorder:
                try:
                    recorder.stop()
                    logger.info("RealtimeSTT recorder stopped.")
                except Exception as e:
                    logger.error(f"Exception during recorder.stop(): {e}", exc_info=True)


    def decode_and_resample(audio_data, original_sample_rate, target_sample_rate=16000):
        """Decodes PCM_S16LE and resamples audio if necessary."""
        global logger
        try:
            # Assuming audio_data is raw PCM S16LE bytes
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            if original_sample_rate == target_sample_rate:
                return audio_data # No resampling needed

            num_original_samples = len(audio_np)
            if num_original_samples == 0:
                # logger.warning("Empty audio data passed to resample.")
                return audio_data # Return empty if input is empty

            num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)
            if num_target_samples == 0:
                # logger.warning(f"Resampling from {original_sample_rate} to {target_sample_rate} resulted in 0 target samples.")
                return np.array([], dtype=np.int16).tobytes() # Return empty if target is zero

            resampled_audio = resample(audio_np, num_target_samples)
            return resampled_audio.astype(np.int16).tobytes()
        except Exception as e:
            logger.error(f"Error in resampling audio from {original_sample_rate} to {target_sample_rate}: {e}", exc_info=True)
            return audio_data # Fallback to original data on error


    async def audio_handler(websocket):
        """Handles incoming WebSocket connections and audio data."""
        global client_websocket, logger, recorder, recorder_ready, is_running
        logger.info("Client connected to audio_handler.")
        client_websocket = websocket # Store the current client

        try:
            # Wait for the recorder to be ready, with a timeout
            if not recorder_ready.wait(timeout=20): # Timeout for recorder initialization
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
                    # Process binary audio data
                    audio_chunk_from_client = message 
                    # Client should send 16kHz, 16-bit PCM mono. If not, resampling might be needed.
                    # For now, assume it's correct and RealtimeSTT handles internal resampling if its target differs.
                    # RealtimeSTT expects 16kHz.
                    
                    # If client sends a different sample rate, it should be declared or detected.
                    # For HTML client, audio worklet should ensure 16kHz.
                    # Here we assume client_sample_rate IS 16000.
                    client_sample_rate = 16000 

                    if audio_chunk_from_client:
                        # Resample if necessary (e.g. if client couldn't send at 16k)
                        # processed_chunk = decode_and_resample(audio_chunk_from_client, client_sample_rate, 16000)
                        # Since RealtimeSTT internally expects 16kHz and should handle minor deviations,
                        # and our HTML client aims for 16kHz, direct feed might be fine.
                        # However, explicit resampling adds robustness if client audio isn't exactly 16kHz.
                        processed_chunk = audio_chunk_from_client # Assuming client sends 16kHz or RealtimeSTT handles it.

                        if recorder:
                            recorder.feed_audio(processed_chunk)
                        else:
                            logger.warning("Recorder not available to feed audio.")
                    else:
                        logger.warning("Received empty audio chunk from client.")
                
                elif isinstance(message, str):
                    # Handle text messages (e.g., commands or metadata)
                    logger.info(f"Received text message from client: {message}")
                    try:
                        data = json.loads(message)
                        if data.get("type") == "control" and data.get("command") == "stop_processing":
                            logger.info("Client requested to stop processing (example command).")
                            # Implement logic if needed, e.g., force finalize sentence
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from text message: {message}")
                    # pass # Ignore non-binary messages for now or handle commands

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Client disconnected gracefully.")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Client connection closed with error: {e}")
        except Exception as e:
            logger.error(f"Error in websocket audio_handler: {e}", exc_info=True)
        finally:
            logger.info("Client session ended.")
            if client_websocket == websocket: # Clear only if it's the active one
                client_websocket = None


    async def main_server_logic():
        """Initializes and runs the main WebSocket server and recorder thread."""
        global main_loop, is_running, logger, recorder_thread
        main_loop = asyncio.get_running_loop() # Get the current event loop

        # Start the recorder thread
        recorder_thread = threading.Thread(target=run_recorder, daemon=True)
        recorder_thread.start()

        # Wait for recorder to initialize, with a longer timeout for model loading
        initialization_timeout = 90 # seconds (e.g. for large-v2/v3 model download & load)
        if not recorder_ready.wait(timeout=initialization_timeout):
            logger.error(f"RealtimeSTT recorder failed to initialize in {initialization_timeout}s. Server not starting.")
            is_running = False # Signal threads to stop
            if recorder_thread.is_alive():
                recorder_thread.join(timeout=5) # Attempt to clean up thread
            return # Exit if recorder fails

        server_host = "0.0.0.0"
        server_port = 7860 # Ensure this matches Dockerfile EXPOSE
        logger.info(f"WebSocket server starting, listening on ws://{server_host}:{server_port}")
        
        stop_event = asyncio.Event() # For graceful shutdown signaling

        try:
            # Setup WebSocket server with ping/pong for keepalive
            async with websockets.serve(
                audio_handler,
                server_host,
                server_port,
                max_size=None, # Allow large messages if needed, though audio chunks should be small
                ping_interval=20, # Send pings every 20 seconds
                ping_timeout=20   # Timeout if pong not received in 20 seconds
            ):
                await stop_event.wait() # Keep server running until stop_event is set
        except OSError as e:
            logger.error(f"OSError starting server (Port {server_port} likely in use or permission issue): {e}")
            is_running = False
        except Exception as e:
            logger.error(f"Main server logic encountered an unexpected error: {e}", exc_info=True)
            is_running = False
        finally:
            logger.info("Server shutdown sequence initiated.")
            is_running = False # Ensure all loops and threads know to stop
            stop_event.set() # Signal the server loop to exit
            if recorder_thread and recorder_thread.is_alive():
                logger.info("Waiting for recorder thread to join...")
                recorder_thread.join(timeout=10) # Give time for graceful shutdown
                if recorder_thread.is_alive():
                    logger.warning("Recorder thread did not join in time.")
            logger.info("Main server logic finished.")

    try:
        asyncio.run(main_server_logic())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down application...")
    finally:
        is_running = False # Final ensure flag is set for any lingering checks
        logger.info("Application shutdown complete.")
