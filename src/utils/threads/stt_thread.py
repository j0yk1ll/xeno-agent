import logging
import queue
import threading
import time

from src.utils.audio_recorder import AudioRecorder
from src.utils.stt import STT

def stt_thread(
    is_recording_event: threading.Event,
    shutdown_event: threading.Event,
    outbound_queue: queue.Queue,
    vad_threshold: float = 0.75,
    sample_rate: int = 16000,
    silence_duration: float = 2.0,
    pre_speech_length: float = 0.5,
):
    """
    Runs the STT processing loop in a separate thread, controlled by events.

    Args:
        is_recording_event (threading.Event): Event indicating if recording should be active.
        shutdown_event (threading.Event): Event to signal thread shutdown.
        outbound_queue (queue.Queue): Queue to send complete transcriptions.
        transcription_model (str, optional): Whisper model name/path.
        vad_model_repo (str, optional): Silero VAD model repo.
        vad_threshold (float, optional): VAD probability threshold.
        sample_rate (int, optional): Audio sample rate.
        silence_duration (float, optional): Silence duration to finalize transcription.
        pre_speech_length (float, optional): Pre-speech buffer length.
    """
    logger = logging.getLogger(__name__)
    logger.info("STT thread started.")

    # Initialize AudioRecorder
    audio_recorder = AudioRecorder(
        sample_rate=sample_rate,
        chunk_size=512,
        queue_maxsize=100,
    )

    # Define the callback to handle complete transcriptions
    def on_complete_transcription(text: str):
        try:
            outbound_queue.put(text, timeout=0.1)
            logger.debug(f"Transcription sent to outbound queue: {text}")
        except queue.Full:
            logger.warning("Outbound queue is full. Dropping transcription.")

    # Initialize STT
    try:
        stt = STT(
            vad_threshold=vad_threshold,
            sample_rate=sample_rate,
            silence_duration=silence_duration,
            pre_speech_length=pre_speech_length,
            on_complete_transcription=on_complete_transcription,
        )
    except Exception as e:
        logger.error(f"Failed to initialize STT: {e}")
        return

    is_recording = False

    try:
        while not shutdown_event.is_set():
            # Handle shutdown
            if shutdown_event.is_set():
                logger.info("Shutdown event detected.")
                break

            # Handle recording state
            if is_recording_event.is_set():
                if not is_recording:
                    audio_recorder.start_recording()
                    is_recording = True
                    logger.info("Recording started.")
            else:
                if is_recording:
                    audio_recorder.stop_recording()
                    is_recording = False
                    logger.info("Recording stopped.")

            # Process audio data
            try:
                audio_data = audio_recorder.audio_queue.get(timeout=0.1)  # 100ms timeout
                if is_recording:
                    logger.debug(f"Processing audio data of length {len(audio_data)} bytes.")
                else:
                    logger.debug(f"Processing remaining audio data of length {len(audio_data)} bytes.")
                stt.process_audio(audio_data)
                audio_recorder.audio_queue.task_done()
            except queue.Empty:
                pass  # No audio data received
            except Exception as e:
                logger.error(f"Error processing audio data: {e}", exc_info=True)

            # Sleep briefly to prevent tight loop
            time.sleep(0.01)

    except Exception as e:
        logger.error(f"Exception in STT thread: {e}", exc_info=True)

    finally:
        # Ensure that recording is stopped
        if is_recording:
            audio_recorder.stop_recording()
            logger.info("Recording stopped during shutdown.")

        logger.info("STT thread shutting down.")
