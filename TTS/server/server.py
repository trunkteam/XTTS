#!flask/bin/python
import argparse
import io
import json
import os
import sys
import uuid
import torch
import torchaudio.functional as taf
from pydub import AudioSegment
from pathlib import Path
from threading import Lock
from typing import Union
from urllib.parse import parse_qs
import time

from flask import Flask, render_template, render_template_string, request, send_file

from typing import List
import numpy as np
from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from tokenizers import Tokenizer

from TTS.tts.utils.text.universal_normalizer import MultilingualNormalizer

normalizer = MultilingualNormalizer()


def resample_and_save_as_mp3(input_array, input_sample_rate, output_mp3_path, target_sample_rate=44100):
    audio_segment = AudioSegment(
        input_array.tobytes(),
        sample_width=input_array.dtype.itemsize,
        frame_rate=input_sample_rate,
        channels=1
    )

    audio_segment = audio_segment.set_frame_rate(target_sample_rate)
    audio_segment.export(output_mp3_path, format="mp3")


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    return parser


# parse the args
args = create_argparser().parse_args()

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

if args.list_models:
    manager.list_models()
    sys.exit()

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit()

# CASE2: load pre-trained model paths
if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

# CASE3: set custom model paths
if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

# load models
config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
model.gpt.init_gpt_for_inference(kv_cache=True, use_deepspeed=False)
model.gpt.eval()
model = model.cuda()
model.tokenizer.preprocess = None
model.tokenizer.tokenizer = Tokenizer.from_file(f"/data/asr/workspace/audio/xtts/models/vocab_v1/bpe_tokenizer.json")

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=speakers_file_path,
    tts_languages_file=None,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=args.use_cuda,
)

app = Flask(__name__)


def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        show_details=args.show_details,
        use_multi_speaker=False,
        use_multi_language=False,
        speaker_ids=None,
        language_ids=None,
        use_gst=False,
    )


@app.route("/details")
def details():
    if args.config_path is not None and os.path.isfile(args.config_path):
        model_config = load_config(args.config_path)
    else:
        if args.model_name is not None:
            model_config = load_config(config_path)

    if args.vocoder_config_path is not None and os.path.isfile(args.vocoder_config_path):
        vocoder_config = load_config(args.vocoder_config_path)
    else:
        if args.vocoder_name is not None:
            vocoder_config = load_config(vocoder_config_path)
        else:
            vocoder_config = None

    return render_template(
        "details.html",
        show_details=args.show_details,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args.__dict__,
    )


lock = Lock()


@app.route("/api/tts", methods=["GET"])
def tts():
    with lock:
        text = request.args.get("text")
        speaker_idx = request.args.get("speaker_id", "")
        language_idx = request.args.get("language_id", "")
        style_wav = request.args.get("style_wav", "")
        speaker_wav = request.args.get("speaker_wav", None)
        temperature = float(request.args.get("temperature", 0.7))
        inference_noise_scale = float(request.args.get("inference_noise_scale", 0.667))
        if speaker_wav == "":
            speaker_wav = None

        style_wav = style_wav_uri_to_dict(style_wav)
        #diacritized_text = diacrtize(text)
        print(f" > Model input before normalization: {text}")
        og_text = text
        text = normalizer.clean_text(text)
        text = normalizer.normalize(text)

        if text.strip() == "":
            text = og_text
        #diacritized_text = text
        print(f" > Model input: {text}")
        print(f" > Speaker Idx: {speaker_idx}")
        print(f" > Language Idx: {language_idx}")
        print(f" > Speaker wav: {speaker_wav}")
        print(f" > Inference noise scale: {inference_noise_scale}")
        print(f" > Temperature: {temperature}")

        #wavs = model.full_inference(
        #    text,
        #    speaker_wav,
        #    "ar",
        #    temperature=temperature,  # Add custom parameters here
        #    gpt_cond_len=3
        #)
        config.temperature = temperature
        wavs = model.synthesize(text, config, speaker_wav, "en", gpt_cond_len=6)

        t0 = time.time()
        #gpt_cond_latent, _, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)
        #chunks = model.inference_stream(text, "en", gpt_cond_latent, speaker_embedding)

        #wav_chuncks = []
        #for i, chunk in enumerate(chunks):
        #    if i == 0:
        #        print(f"Time to first chunck: {time.time() - t0}")
        #    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        #    wav_chuncks.append(chunk)
        #wav = torch.cat(wav_chuncks, dim=0)


        out = io.BytesIO()
        # synthesizer.save_wav(wavs, out)
        wav = np.array(wavs["wav"])
        #wav = np.array(wav.squeeze().unsqueeze(0).cpu())
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wav_norm = wav_norm.astype(np.int16)
        resample_and_save_as_mp3(input_array=wav_norm, input_sample_rate=24000, output_mp3_path=out,
                                 target_sample_rate=44100)
        try:
            if speaker_wav:
                os.remove(speaker_wav)
        except OSError:
            pass
    return send_file(out, mimetype="audio/wav")


# Basic MaryTTS compatibility layer


@app.route("/locales", methods=["GET"])
def mary_tts_api_locales():
    """MaryTTS-compatible /locales endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string("{{ locale }}\n", locale=model_details[1])


@app.route("/voices", methods=["GET"])
def mary_tts_api_voices():
    """MaryTTS-compatible /voices endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string(
        "{{ name }} {{ locale }} {{ gender }}\n", name=model_details[3], locale=model_details[1], gender="u"
    )


@app.route("/process", methods=["GET", "POST"])
def mary_tts_api_process():
    """MaryTTS-compatible /process endpoint"""
    with lock:
        if request.method == "POST":
            data = parse_qs(request.get_data(as_text=True))
            # NOTE: we ignore param. LOCALE and VOICE for now since we have only one active model
            text = data.get("INPUT_TEXT", [""])[0]
        else:
            text = request.args.get("INPUT_TEXT", "")
        print(f" > Model input: {text}")
        wavs = synthesizer.tts(text)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == 'POST':
        file_path = f"/tmp/{str(uuid.uuid4())}.wav"
        f = request.files['file']
        f.save(file_path)
        return {"path": file_path}


def main():
    app.run(debug=args.debug, host="::", port=args.port)


if __name__ == "__main__":
    main()

