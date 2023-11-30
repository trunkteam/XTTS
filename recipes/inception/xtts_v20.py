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

EXP_ID = "v20"
BASE_PATH = "/data/asr/workspace/audio/xtts"
DATA_PATH = "/data/asr/workspace/audio/tts"
EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/xtts/{EXP_ID}")

# Set here the path that the checkpoints will be saved. Default: ./run/training/

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 12  # set here the batch size
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
    # get_dataset(manifest_train="data/tts2/manifest/ar/ar_qu_v1/22k/manifest_dur.json",
    #             manifest_eval="data/tts2/manifest/ar/ar_qu_v1/22k/manifest_eval_dur_2.json",
    #             d_name="ar_qu_v1",
    #             lang="ar"),
    get_dataset(manifest_train="data/tts2/manifest/ar/ar_se_v1/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/ar/ar_se_v1/22k/manifest_eval_dur.json",
                d_name="ar_se_v1",
                lang="ar"),
    get_dataset(manifest_train="data/tts/ar/spotify/manifest/manifest_spk_cluster_diac_22050sr.json",
                manifest_eval="data/tts/ar/spotify/manifest/manifest_spk_cluster_diac_22050sr_eval.json",
                d_name="ar_spotify_v1",
                lang="ar"),
    # get_dataset(manifest_train="data/tts2/manifest/en/en_az_gen_v1/22k/manifest_dur.json",
    #             manifest_eval="data/tts2/manifest/en/en_az_gen_v1/22k/manifest_eval_dur.json",
    #             d_name="en_az_gen_v1",
    #             lang="en"),
    get_dataset(manifest_train="data/tts2/manifest/en/en_el_gen_v2/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/en/en_el_gen_v2/22k/manifest_eval_dur.json",
                d_name="en_el_gen_v2",
                lang="en"),
    get_dataset(manifest_train="data/tts2/manifest/en/en_el_gen_v3/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/en/en_el_gen_v3/22k/manifest_eval_dur.json",
                d_name="en_el_gen_v3",
                lang="en"),
    get_dataset(manifest_train="data/tts2/manifest/en/en_se_v1/22k/manifest_dur.json",
                manifest_eval="data/tts2/manifest/en/en_se_v1/22k/manifest_eval_dur.json",
                d_name="en_se_v1",
                lang="en"),
    get_dataset(manifest_train="data/tts/en/spotify/manifest/manifest_spk_cluster_22050sr.json",
                manifest_eval="data/tts/en/spotify/manifest/manifest_spk_cluster_22050sr_eval.json",
                d_name="en_spotify_v1",
                lang="en"),
    get_dataset(manifest_train="data/audio/multi_lang/char/manifest_org_hi_ns_clean.json",
                manifest_eval="data/audio/multi_lang/char/manifest_org_hi_ns_clean_eval.json",
                d_name="hi_v1",
                lang="hi"),
    get_dataset(manifest_train="data/hindi_vo/manifests/manifest_22050sr.json",
                manifest_eval="data/hindi_vo/manifests/manifest_22050sr_eval.json",
                d_name="hi_v2",
                lang="hi"),
    get_dataset(manifest_train="data/arabic/sheikh_zayed_combined/manifest_combined.json",
                manifest_eval="data/arabic/sheikh_zayed_combined/manifest_combined_eval.json",
                d_name="sh_ar_v4",
                lang="ar"),
    get_dataset(manifest_train="data/arabic/sheikh_zayed_combined/manifest_shz_5.json",
                manifest_eval="data/arabic/sheikh_zayed_combined/manifest_shz_5_eval.json",
                d_name="sh_ar_v5",
                lang="ar"),
    get_dataset(manifest_train="data/arabic/shz_v2/manifests/manifest_22050sr.json",
                manifest_eval="data/arabic/shz_v2/manifests/manifest_22050sr_eval.json",
                d_name="sh_ar_v6",
                lang="ar"),
    get_dataset(manifest_train="data/arabic/shz_v7/manifest_22050sr.json",
                manifest_eval="data/arabic/shz_v7/manifest_22050sr_eval.json",
                d_name="sh_ar_v7",
                lang="ar"),
    get_dataset(manifest_train="data/audio/multi_lang/char/manifest_org_en_ns_clean_ut_500.json",
                manifest_eval="data/audio/multi_lang/char/manifest_org_en_ns_clean_ut_500_eval.json",
                d_name="org_libri",
                lang="en"),
    get_dataset(manifest_train="data/english/lewis_hamilton_hq/manifest_greedy.json",
                manifest_eval="data/english/lewis_hamilton_hq/manifest_greedy_eval.json",
                d_name="lewis_hamilton",
                lang="en"),
    get_dataset(manifest_train="data/english/andrew/manifests/manifest_22050sr.json",
                manifest_eval="data/english/andrew/manifests/manifest_22050sr_eval.json",
                d_name="andrew_v1",
                lang="en"),
    get_dataset(manifest_train="data/english/andrew/v2/manifests/manifest_22050sr.json",
                manifest_eval="data/english/andrew/v2/manifests/manifest_22050sr_eval.json",
                d_name="andrew_v2",
                lang="en"),
    get_dataset(manifest_train="data/english/peng_xiao_v2/manifests/manifest_22050sr.json",
                manifest_eval="data/english/peng_xiao_v2/manifests/manifest_22050sr_eval.json",
                d_name="peng_xiao",
                lang="en"),
    # get_dataset(manifest_train="data/english/xtts_gen_v3/manifest_22050sr_partial.json",
    #             manifest_eval="data/english/xtts_gen_v3/manifest_22050sr_partial_eval.json",
    #             d_name="xtts_gen_v3_partial",
    #             lang="en"),
    get_dataset(manifest_train="data/tts/en/v7/limmits/manifest_22k.json",
                manifest_eval="data/tts/en/v7/limmits/manifest_22k_eval.json",
                d_name="en_limmits",
                lang="en"),
    get_dataset(manifest_train="data/tts/hi/v7/limmits/manifest_22k.json",
                manifest_eval="data/tts/hi/v7/limmits/manifest_22k_eval.json",
                d_name="hi_limmits",
                lang="hi"),
    get_dataset(manifest_train="data/audio/multi_lang/char/manifest_org_ml_ns_clean.json",
                manifest_eval="data/audio/multi_lang/char/manifest_org_ml_ns_clean_eval.json",
                d_name="ml_ns",
                lang="ml"),
    get_dataset(manifest_train="data/malayalam/huggingface/GMaSC/manifest_22050sr.json",
                manifest_eval="data/malayalam/huggingface/GMaSC/manifest_22050sr_eval.json",
                d_name="GMaSC",
                lang="ml"),
    get_dataset(manifest_train="data/malayalam/huggingface/IMaSC/manifest_22050sr.json",
                manifest_eval="data/malayalam/huggingface/IMaSC/manifest_22050sr_eval.json",
                d_name="IMaSC",
                lang="ml"),
    get_dataset(manifest_train="data/malayalam/huggingface/msc/manifest_22050sr.json",
                manifest_eval="data/malayalam/huggingface/msc/manifest_22050sr_eval.json",
                d_name="msc",
                lang="ml"),
    get_dataset(manifest_train="data/malayalam/huggingface/ulca_ml/manifest_22050sr.json",
                manifest_eval="data/malayalam/huggingface/ulca_ml/manifest_22050sr_eval.json",
                d_name="ulca_ml",
                lang="ml"),
]

# Define the path where XTTS v2.0 files will be downloaded
CHECKPOINTS_OUT_PATH = f"{BASE_PATH}/models/v2.0/"
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Download XTTS v2 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
# TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab_hindi.json")  # vocab.json file
TOKENIZER_FILE = "/data/asr/workspace/audio/xtts/models/vocab_v2/bpe_tokenizer.json"
XTTS_CHECKPOINT ="/data/asr/workspace/audio/xtts/expmt/xtts/v19/Inception-XTTS-November-17-2023_08+28PM-04901fb2/best_model_756162.pth"  # model.pth file

# download XTTS v1.1 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v1.1 files!")
    ModelManager._download_model_files([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH,
                                       progress_bar=True)

# Training sentences generations
SPEAKER_REFERENCE = (
    "/data/asr/workspace/audio/tts/data/tts/ar/spotify/wav/22k/33cade80-c9a2-4947-8361-a3ee698363a0.wav",
    "/data/asr/workspace/audio/tts/data/tts/ar/spotify/wav/22k/95771e82-ee14-48d4-b51c-6317be78d300.wav",
    "/data/asr/workspace/audio/tts/data/tts/ar/spotify/wav/22k/610321f5-4aeb-49e6-810e-81504cd28d6f.wav",
    "/data/asr/workspace/audio/tts/data/tts/ar/spotify/wav/22k/475af21a-8f9c-4fd1-b5b8-0b47833fd5f9.wav",
    "/data/asr/workspace/audio/tts/data/tts/ar/spotify/wav/22k/4ba83a77-5f0e-43ac-907b-b23f94dd7edf.wav",
    "/data/asr/workspace/audio/tts/data/tts/en/spotify/wav/22k/9e639152-bf6c-4c6e-a197-f95cfb9ad5ab.wav",
    "/data/asr/workspace/audio/tts/data/tts/en/spotify/wav/22k/77d02aa4-eef3-44c5-82fb-6e4a7c274bf2.wav",
    "/data/asr/workspace/audio/tts/data/tts/en/spotify/wav/22k/caafcf22-e62b-4ee5-89bd-195cd3912e50.wav",
    f"/data/asr/workspace/audio/tts/data/tts/ar/wav/v3/spk_ar_el_v3_1/22k/fcddf3c8-8a3b-4528-a72a-6defa6ecab3f.wav",
    f"/data/asr/workspace/audio/tts/data/tts/en/v5/gen/Abeo_En_Ng/wav/22k/11897989-6c98-45a0-96dc-bf776c26687a.wav",
    f"/data/asr/workspace/audio/tts/data/tts/en/wav/v3/spk_en_el_v3_0/22k/80dc7c93-ae14-45bb-8011-c4d16aefa06f.wav",
    f"/data/asr/workspace/audio/tts/data/tts/en/wav/v3/spk_en_el_v3_15/22k/213f2e64-97e3-4f4d-ad20-590eff14e18c.wav",
    f"/data/asr/workspace/audio/tts/data/hindi/wav_22k/spk_g_0/b2de04e9-efbe-41e9-a3e1-ec5c8e496665_16b.wav",
    f"/data/asr/workspace/audio/tts/data/arabic/shz_manual/wavs_22k/a05ba08b-5c36-437c-ba4c-876b0e5c3a8a_audio_441.wav"
)


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
        gpt_max_prompt_tokens=140,
        gpt_layers=48,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
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
        save_n_checkpoints=2,
        save_checkpoints=True,
        save_all_best=True,
        languages=[
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh-cn",
            "hu",
            "ko",
            "ja",
            "hi",
            "ml"
        ],
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "Alright folks, you're tuned into ninety nine point nine Dubai's A D D C Morning Mania with RJ Danny and RJ Sam. We're here to make your morning traffic sound like a lullaby...or a comedy show. Whichever you prefer!",
                "speaker_wav": SPEAKER_REFERENCE[0],
                "language": "en",
            },
            {
                "text": "Umm, so, like, I don't what you're, you know, totally getting at, but, it's, uh, kinda hard to, erm, figure out without, like, more deets, LOL.",
                "speaker_wav": SPEAKER_REFERENCE[1],
                "language": "en",
            },
            {
                "text": "أَنَا أَنَااظِرْ الْقِمْمَّ. وَأَشُوفْ اِلْإِمَارَاتْ وِصْلَتْ. زَايِدْ، Gَالْكُمْ يِبَا الْخِيرْ لَهَذَا الشَّعِبْ. تَرَكْتِ لْكُمْ خَلِيفَهْ مِنْ بَعْدِي، وِالْيُومِ مْحَمَّدْ يِكَمِّلْ الْمَسِيرَ.",
                "speaker_wav": SPEAKER_REFERENCE[2],
                "language": "ar",
            },
            {
                "text": "وْحَيِّيتُو ذِكْرَايْ، بْعَامْ زَايِدْ. اِللِّي وِضَحْ لِي، إِنِّي لِينَ الْحِينْ حَيّْ، بِقْلُوبْكُمْ وْذَاكْرِتْكُمْ. أَنَا أَفْخَرْ بِكُمْ وَاحِدْ وَاحِدْ، بِاللِّي حَقَّقْتُوهْ.",
                "speaker_wav": SPEAKER_REFERENCE[3],
                "language": "ar",
            },
            {
                "text": 'जहाँ रात के आसपास की दुकानें रौशनी की तलाश में होती हैं और जीवन अपने रंग-रूप में बेहद मजेदार होता है। यहाँ का भोजन तो मस्ती और आनंद का नाम है, और इस शहर का रित्म बिल्कुल अपना है।',
                "speaker_wav": SPEAKER_REFERENCE[6],
                "language": "hi"
            },
            {
                "text": 'सुप्रभात! सर्दी में खुलकर आसमान में बिकती धूप के ताप से जगमगाता परिदेश जैसा लगता है।',
                "speaker_wav": SPEAKER_REFERENCE[7],
                "language": "hi"
            },
            {
                "text": 'കേരളത്തിനുള്ള ഓണസമ്മാനമായി രണ്ടാം വന്ദേഭാരത് ട്രെയിൻ പാലക്കാട് ഡിവിഷന് അനുവദിച്ചതായി റിപ്പോർട്ട്.',
                "speaker_wav": SPEAKER_REFERENCE[8],
                "language": "ml"
            },
            {
                "text": 'ന്യൂഡൽഹി. ഇന്ത്യയുടെ പ്രദേശങ്ങൾ ഉൾപ്പെടുത്തി ചൈന ഭൂപടം പ്രസിദ്ധീകരിച്ചത് ഗൗരവമേറിയതെന്ന് മുതിർന്ന കോൺഗ്രസ് നേതാവ് രാഹുൽ ഗാന്ധി. ഈ വിഷയത്തിൽ പ്രധാനമന്ത്രി നരേന്ദ്ര മോദി സംസാരിക്കണമെന്നും രാഹുൽ‌ ആവശ്യപ്പെട്ടു.',
                "speaker_wav": SPEAKER_REFERENCE[9],
                "language": "ml"
            },
            {
                "text": 'ബെംഗളൂരു. ചന്ദ്രോപരിതലത്തിൽ സൾഫറിന്റെ സാന്നിധ്യം സ്ഥിരീകരിച്ച് ചന്ദ്രയാൻ ദൗത്യം. ദൗത്യത്തിലെ വിക്രം ലാൻഡറിൽനിന്നു പുറത്തിറങ്ങിയ പ്രഗ്യാൻ റോവറിലുള്ള ലേസർ ഇൻഡ്യൂസ്ഡ് ബ്രേക്ഡൗൺ സ്പെക്ട്രോസ്കോപ് (ലിബ്സ്) എന്ന ശാസ്ത്രീയ ഉപകരണമാണ് ചന്ദ്രന്റെ ദക്ഷിണ ധ്രുവത്തിൽ സൾഫറിന്റെ സാന്നിധ്യം സ്ഥിരീകരിച്ചത്.',
                "speaker_wav": SPEAKER_REFERENCE[10],
                "language": "ml"
            },
        ]
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
