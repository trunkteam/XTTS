import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# Logging parameters
RUN_NAME = "Inception-XTTS"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

EXP_ID = "v5"
BASE_PATH = "/data/asr/workspace/audio/xtts"
DATA_PATH = "/data/asr/workspace/audio/tts"
EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/xtts/{EXP_ID}")

# Set here the path that the checkpoints will be saved. Default: ./run/training/

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 3  # set here the batch size
GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps


# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.


def get_dataset(manifest_train: str, manifest_eval: str, d_name: str, lang: str = "ar", base_path=DATA_PATH,
                data_path=DATA_PATH):
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"{lang}_{d_name}",
        meta_file_train=os.path.join(data_path, manifest_train),
        meta_file_val=os.path.join(data_path, manifest_eval),
        path=base_path,
        language=lang,
    )


DATASETS_CONFIG_LIST = [
    get_dataset(manifest_train="data/tts2/manifest/ar/ar_el_gen_v2/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/ar/ar_el_gen_v2/22k/manifest_eval_dur.json",
                d_name="ar_el_gen_v2",
                lang="ar"),
    get_dataset(manifest_train="data/tts2/manifest/ar/ar_el_gen_v3/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/ar/ar_el_gen_v3/22k/manifest_eval_dur.json",
                d_name="ar_el_gen_v3",
                lang="ar"),
    get_dataset(manifest_train="data/tts2/manifest/ar/ar_el_gen_v5/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/ar/ar_el_gen_v5/22k/manifest_eval_dur.json",
                d_name="ar_el_gen_v5",
                lang="ar"),
    get_dataset(manifest_train="data/tts2/manifest/ar/ar_qu_v1/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/ar/ar_qu_v1/22k/manifest_eval_dur.json",
                d_name="ar_qu_v1",
                lang="ar"),
    # get_dataset(manifest_train="data/tts2/manifest/ar/ar_se_v1/22k/manifest_dur.json",
    #             manifest_eval="data/tts2/manifest/ar/ar_se_v1/22k/manifest_eval_dur.json",
    #             d_name="ar_se_v1",
    #             lang="ar"),
    get_dataset(manifest_train="data/tts2/manifest/en/en_az_gen_v1/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/en/en_az_gen_v1/22k/manifest_eval_dur.json",
                d_name="en_az_gen_v1",
                lang="en"),
    get_dataset(manifest_train="data/tts2/manifest/en/en_el_gen_v2/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/en/en_el_gen_v2/22k/manifest_eval_dur.json",
                d_name="en_el_gen_v2",
                lang="en"),
    get_dataset(manifest_train="data/tts2/manifest/en/en_el_gen_v3/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/en/en_el_gen_v3/22k/manifest_eval_dur.json",
                d_name="en_el_gen_v3",
                lang="en"),
    # get_dataset(manifest_train="data/tts2/manifest/en/en_se_v1/22k/manifest_dur.json",
    #             manifest_eval="data/tts2/manifest/en/en_se_v1/22k/manifest_eval_dur.json",
    #             d_name="en_se_v1",
    #             lang="en"),
    get_dataset(manifest_train="data/audio/multi_lang/char/manifest_org_hi_ns_clean.json",
                manifest_eval="data/audio/multi_lang/char/manifest_org_hi_ns_clean_eval.json",
                d_name="hi_v1",
                lang="hi"),
    get_dataset(manifest_train="data/hindi_vo/manifests/manifest_22050sr.json",
                manifest_eval="data/hindi_vo/manifests/manifest_22050sr_eval.json",
                d_name="hi_v2",
                lang="hi"),
]

# Define the path where XTTS v1.1.1 files will be downloaded
CHECKPOINTS_OUT_PATH = f"{BASE_PATH}/models/pretrained/"
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.1/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.1/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Download XTTS v1.1 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.1/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.1/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab_hindi.json")  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")  # model.pth file

# download XTTS v1.1 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v1.1 files!")
    ModelManager._download_model_files([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH,
                                       progress_bar=True)

# Training sentences generations
SPEAKER_REFERENCE = (
    f"{DATA_PATH}/data/tts/ar/wav/spk_el_8/22k/3df9454d-32e8-4a5f-8303-21561d666fd1.wav",  # speaker reference to be used in training test sentences
    f"{DATA_PATH}/data/tts/ar/wav/spk_el_10/22k/dd502bb5-f19c-4193-8dfb-aab216eaf6cd.wav",
    f"{DATA_PATH}/data/tts/ar/wav/spk_el_15/22k/e3e429f1-cd3a-4b86-b4af-56fa1eed1d9f.wav",
    f"{DATA_PATH}/data/tts/en/v5/gen/Abeo_En_Ng/wav/22k/11897989-6c98-45a0-96dc-bf776c26687a.wav",
    f"{DATA_PATH}/data/tts/en/v5/gen/Abbi_En_Gb/wav/22k/1fc3cb86-27d9-464a-be72-2cfd6799c44c.wav",
    f"{DATA_PATH}/data/tts/en/v5/gen/Elimu_En_Tz/wav/22k/49840458-d09a-4f79-9c65-449e872eb064.wav",
    f"{DATA_PATH}/data/hindi/wav_22k/spk_g_0/b2de04e9-efbe-41e9-a3e1-ec5c8e496665_16b.wav"
)


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=220500,  # 10 secs
        min_conditioning_length=44100,  # 2 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~20 seconds
        max_text_length=512,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        # tokenizer_file="/raid/datasets/xtts_models/vocab.json", # vocab path of the model that you want to fine-tune
        # xtts_checkpoint="https://huggingface.co/coqui/XTTS-v1/resolve/hifigan/model.pth",
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=8194,
        gpt_start_audio_token=8192,
        gpt_stop_audio_token=8193,
        use_ne_hifigan=True,  # if it is true it will keep the non-enhanced keys on the output checkpoint
        use_hifigan=True,
        gpt_n_heads=16,
        gpt_layers=36,
        diff_num_layers=16,
        diff_num_heads=24
    )
    # define audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, diffusion_sample_rate=24000, output_sample_rate=24000
    )
    # training parameters config
    config = GPTTrainerConfig(
        epochs=1000,
        output_path=EXPMT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hi"],
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=0.001,  # learning rate
        lr_scheduler=None,
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={},
        test_sentences=[
            {
                "text": "Alright folks, you're tuned into ninety-nine-point-nine Dubai's ADDC Morning Mania with RJ Danny and RJ Sam. We're here to make your morning traffic sound like a lullaby...or a comedy show. Whichever you prefer!",
                "speaker_wav": SPEAKER_REFERENCE[3],
                "language": "en",
            },
            {
                "text": "Umm, so, like, IDK what you're, you know, totally getting at, but TBH, it's, uh, kinda hard to, erm, figure out without, like, more deets, LOL.",
                "speaker_wav": SPEAKER_REFERENCE[1],
                "language": "en",
            },
            {
                "text": "وَكَانْ بِيطَلِعِ الفَلِمْ بَسْ وْقَفْتْ اِيدَهْ فَجَاهْ وَهُو يِشُوفْ حِصَهْ يَالْسِهْ تَظَحَكْ بَعَدْ مَارْكِزُوا عَلِيهَا الْكَامِيرَا عَدَالْ عِنُودْ.",
                "speaker_wav": SPEAKER_REFERENCE[0],
                "language": "ar",
            },
            {
                "text": "هَلْ تَجْعَلُ مَحَطَّةَ التِّلِفْرِيكِ الْجَدِيدَةَ السِّيَاحَةَ فِي جِبَالِ عَجْلُونَ الْأُرْدُنِّيَّةِ صَدِيقَةٌ لِلْبِيئَةِ؟",
                "speaker_wav": SPEAKER_REFERENCE[4],
                "language": "ar",
            },
            {
                "text": 'कहते हैं अगर किसी चीज़ को दिल से चाहो तो पूरी कायनात उसे तुमसे मिलाने की कोशिश में लग जाती है।',
                "speaker_wav": SPEAKER_REFERENCE[6],
                "language": "hi"
            },
            {
                "text": 'दुनिया के सबसे बेहतरीन और मशहूर कलाकार वो लोग होते हैं, जिनकी अपनी एक अदा होती है… वो अदा जो किसी की नकल करने से नहीं आती… वो अदा जो उनके साथ जन्म लेती है।',
                "speaker_wav": SPEAKER_REFERENCE[6],
                "language": "hi"
            },
        ],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=EXPMT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
