import os
import json
from trainer import Trainer, TrainerArgs
import numpy as np

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN
from TTS.config.shared_configs import BaseAudioConfig

EXP_ID = "v154"
EXPMT_PATH = f"/data/asr/workspace/audio/tts/expmt/hifigan/{EXP_ID}"


def generate_data_from_manifest(manifest_path, eval_split_size):
    data = []
    with open(manifest_path) as src_m:
        for line in src_m:
            jd = json.loads(line)
            if "/data/asr/workspace/audio/tts" not in jd["audio_filepath"]:
                jd["audio_filepath"] = os.path.join("/data/asr/workspace/audio/tts", jd["audio_filepath"])

            if "/data/asr/workspace/audio/tts" not in jd["mel_file_path"]:
                jd["mel_file_path"] = os.path.join("/data/asr/workspace/audio/tts", jd["mel_file_path"])

            if os.path.exists(jd["audio_filepath"]) and os.path.exists(jd["mel_file_path"]):
                data.append([jd["audio_filepath"], jd["mel_file_path"]])

    print(f"Total samples: {len(data)}")
    np.random.seed(0)
    np.random.shuffle(data)
    return data[:eval_split_size], data[eval_split_size:]


audio_config = BaseAudioConfig(
    sample_rate=44100,
    mel_fmin=0.0,
    mel_fmax=None,
    ref_level_db=20,
    preemphasis=0.0,
    hop_length=512,
    win_length=2048,
    fft_size=2048,
    num_mels=80,
)

config = HifiganConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=10000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=1000,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path="/data/asr/workspace/audio/tts/data/hifigan/manifest_44100sr_2.json",
    output_path=EXPMT_PATH,
    audio=audio_config,
    generator_model_params={'upsample_factors': [8, 8, 4, 2],
                            'upsample_kernel_sizes': [16, 16, 4, 4],
                            'upsample_initial_channel': 512,
                            'resblock_kernel_sizes': [3, 7, 11],
                            'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                            'resblock_type': '1'
                            },
    l1_spec_loss_params={'use_mel': True,
                         'sample_rate': 44100,
                         'n_fft': 2048,
                         'hop_length': 512,
                         'win_length': 2048,
                         'n_mels': 80,
                         'mel_fmin': 0.0,
                         'mel_fmax': None}
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = generate_data_from_manifest(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, EXPMT_PATH, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()

