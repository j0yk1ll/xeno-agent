import logging
import sounddevice as sd
import numpy as np

class AudioPlayer:
    def __init__(self, desired_sample_rate=24000, **kwargs):
        self.desired_sample_rate = desired_sample_rate
        self.output_device, self.sample_rate = self._configure_audio()
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            device=self.output_device,
            channels=1,  # Assuming mono audio; adjust if necessary
            dtype='int16'  # Assuming PCM 16-bit; adjust if necessary
        )
        self.stream.start()
        logging.info("Audio stream started.")

    def _configure_audio(self):
        """
        Configure the audio output device and sample rate based on system settings.
        """
        try:
            default_output_device = sd.default.device[1]  # (input, output)
            if default_output_device is None:
                raise RuntimeError("No default output device set.")

            device_info = sd.query_devices(default_output_device, 'output')
            logging.info(f"Default output device: {device_info['name']}")

            # Check if desired sample rate is supported
            try:
                sd.check_output_settings(device=default_output_device, samplerate=self.desired_sample_rate)
                logging.info(f"Sample rate {self.desired_sample_rate} Hz is supported.")
                return default_output_device, self.desired_sample_rate
            except Exception as e:
                logging.warning(f"Desired sample rate {self.desired_sample_rate} Hz not supported by the default device.")
                # Fallback to common sample rates
                common_sample_rates = [44100, 22050, 48000]
                for rate in common_sample_rates:
                    try:
                        sd.check_output_settings(device=default_output_device, samplerate=rate)
                        logging.info(f"Using fallback sample rate: {rate} Hz.")
                        return default_output_device, rate
                    except:
                        continue
                # If none of the common rates are supported, use device's default
                fallback_rate = int(device_info['default_samplerate'])
                logging.info(f"Using device's default sample rate: {fallback_rate} Hz.")
                return default_output_device, fallback_rate
        except Exception as e:
            logging.error(f"Error configuring audio: {e}")
            raise

    def play_audio_chunk(self, chunk):
        """
        Play a single chunk of PCM audio data.
        """
        try:
            # Assuming chunk is in bytes, convert to numpy array
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            # Ensure the audio data is in the correct shape
            audio_data = audio_data.reshape(-1, 1)  # Mono; adjust channels if necessary
            self.stream.write(audio_data)
            logging.debug("Played a chunk of audio.")
        except Exception as e:
            logging.error(f"Error playing audio chunk: {e}")

    def close(self):
        """
        Close the audio stream.
        """
        self.stream.stop()
        self.stream.close()
        logging.info("Audio stream closed.")