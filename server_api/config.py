import os

MODEL_PATH = os.environ.get("MODEL_PATH", "/data/asr/workspace/audio/xtts/expmt/xtts/v15/Inception-XTTS-November-10-2023_12+39PM-6f1cba2f/best_model_761685.pth")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "/data/asr/workspace/audio/xtts/expmt/xtts/v15/Inception-XTTS-November-10-2023_12+39PM-6f1cba2f/config.json")


DIACRITIZER_MODEL_CONFIG_PATH = os.environ.get("DIACRITIZER_MODEL_CONFIG_PATH", "/data/asr/workspace/audio/xtts/TTS/TTS/tts/utils/text/arabic/deepDiac/configs/config_d3.yaml")
DIACRITIZER_MODEL_TEST_SEGMENT_CONFIG_PATH = os.environ.get("DIACRITIZER_MODEL_TEST_SEGMENT_CONFIG_PATH", "/data/asr/workspace/audio/xtts/TTS/TTS/tts/utils/text/arabic/deepDiac/configs/segment_test.yaml")
DIACRITIZER_WORD_EMBEDDING_PATH = os.environ.get("DIACRITIZER_WORD_EMBEDDING_PATH", "/data/asr/workspace/audio/xtts/TTS/TTS/tts/utils/text/arabic/deepDiac/dataset/oscar_p1/vocab.vec")
DIACRITIZER_CONSTANTS_PATH = os.environ.get("DIACRITIZER_CONSTANTS_PATH", "/data/asr/workspace/audio/xtts/TTS/TTS/tts/utils/text/arabic/deepDiac/dataset/helpers/constants")
DIACRITIZER_MODEL_PATH = os.environ.get("DIACRITIZER_MODEL_PATH", "/data/asr/workspace/audio/xtts/TTS/TTS/tts/utils/text/arabic/deepDiac/models/oscar_p1_v1.1_d3/oscar_p1_v1.1_d3.best.pt")
DIACRITIZER_BASE_PATH = os.environ.get("DIACRITIZER_BASE_PATH", "/data/asr/workspace/audio/xtts/TTS/TTS/tts/utils/text/arabic/deepDiac/dataset/oscar_p1/")
