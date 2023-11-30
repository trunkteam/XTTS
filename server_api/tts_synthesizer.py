import torch
import numpy as np
from TTS.utils.synthesizer import Synthesizer
# from TTS.tts.utils.text.universal_normalizer import MultilingualNormalizer


class TTSSynthesizer:
    def __init__(self, model_path, config_path, use_cuda=torch.cuda.is_available()):
        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            use_cuda=use_cuda
        )
        self.speaker_manager = getattr(self.synthesizer.tts_model, "speaker_manager", None)
        self.model = self.synthesizer.tts_model
        # self.normalizer = MultilingualNormalizer()

    def synthesize(self, text, speaker, speaker_wav=None, length_scale=None, inference_noise_scale=None,
                   style_wav=None, inference_noise_scale_dp=None):
        if length_scale:
            self.synthesizer.tts_model.length_scale = length_scale

        if inference_noise_scale:
            self.synthesizer.tts_model.inference_noise_scale = inference_noise_scale

        if inference_noise_scale_dp:
            self.synthesizer.tts_model.inference_noise_scale_dp = inference_noise_scale_dp

        if style_wav == "":
            style_wav = None
        if speaker_wav == "":
            speaker_wav = None

        # text = self.normalizer.clean_text(text)
        # text = self.normalizer.normalize(text)

        wavs = self.synthesizer.tts(text, speaker_name=speaker, style_wav=style_wav, speaker_wav=speaker_wav)
        wav = np.array(wavs)
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        return wav_norm.astype(np.int16)
