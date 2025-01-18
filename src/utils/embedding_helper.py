from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
import torchaudio
from io import BytesIO

from src.utils.audioclip import AudioCLIP

# Define the xeno directory
xeno_models_dir = Path.home() / ".xeno" / "models"
xeno_audioclip_models_dir = xeno_models_dir / "audioclip"

class ToTensor1D(torchvision.transforms.ToTensor):
    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])
        return tensor_2d.squeeze_(0)

class EmbeddingHelper:
    def __init__(
        self,
        model_path: str = str(xeno_audioclip_models_dir / "AudioCLIP-Full-Training.pt"),
        sample_rate: int = 16000,
        image_size: int = 224,
        image_mean: tuple = (0.48145466, 0.4578275, 0.40821073),
        image_std: tuple = (0.26862954, 0.26130258, 0.27577711),
        device: str = None
    ):
        """
        Initializes the AudioCLIP model and defines necessary transformations.
        """
        # Device configuration
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the pre-trained AudioCLIP model
        self.model = AudioCLIP(pretrained=str(model_path)).to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

        # Audio transformation
        self.audio_transforms = ToTensor1D()

        # Image transformation
        self.image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size, interpolation=Image.BICUBIC),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.Normalize(image_mean, image_std)
        ])

        # Constants
        self.sample_rate = sample_rate

        # Obtain logit scales (not strictly required for embedding, but OK to keep)
        self.scale_audio_image = torch.clamp(self.model.logit_scale_ai.exp(), min=1.0, max=100.0).to(self.device)
        self.scale_audio_text = torch.clamp(self.model.logit_scale_at.exp(), min=1.0, max=100.0).to(self.device)
        self.scale_image_text = torch.clamp(self.model.logit_scale.exp(), min=1.0, max=100.0).to(self.device)

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from the specified file path using Pillow.
        """
        return Image.open(image_path).convert('RGB')

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Loads an audio file, resamples it if necessary, and returns as a tensor.
        """
        waveform, original_sample_rate = torchaudio.load(audio_path)  # [channels, samples]
        
        # If stereo, convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if sample rates differ
        if original_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Make it 1D
        waveform = waveform.squeeze(0)  # shape: [samples]
        return waveform

    def create_text_embedding(self, text: str) -> torch.Tensor:
        """
        Creates normalized text embedding from a single text string.
        """
        # Prepare the input as a list containing the single text
        text_inputs = [text]

        # Pass the input through the model and extract features
        _, _, text_features = self.model(text=text_inputs)[0][0]

        # Normalize the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Return the normalized features to the specified device
        return text_features.to(self.device)

    def create_image_embedding(self, image_bytes: bytes) -> torch.Tensor:
        """
        Creates a normalized image embedding from raw in-memory image bytes. (ONLY PNG)
        """
        # Convert bytes to a PIL image
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        # Apply transforms
        image_tensor = self.image_transforms(image).unsqueeze(0).to(self.device)
        # Get image features
        _, image_features, _ = self.model(image=image_tensor)[0][0]
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze(0)

    def create_audio_embedding(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Creates a normalized audio embedding from raw in-memory audio bytes. (only WAV)
        """
        with BytesIO(audio_bytes) as f:
            waveform, original_sample_rate = torchaudio.load(f)

        # If stereo, convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if original_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Make it 1D
        waveform = waveform.squeeze(0)

        # Transform
        audio_tensor = self.audio_transforms(waveform.unsqueeze(0)).unsqueeze(0).to(self.device)  # shape: [1, 1, samples]

        # Get audio features
        audio_features, _, _ = self.model(audio=audio_tensor)[0][0]
        # Normalize
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        return audio_features.squeeze(0)
