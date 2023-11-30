import re
from lingua import Language, LanguageDetectorBuilder
from gruut import sentences
from nemo_text_processing.text_normalization import Normalizer
from TTS.tts.utils.text.arabic.diacritizer_msa import MSADiacritizer
import logging

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("ASR")


class Settings:
    SPACE = " "
    BLANK = ""
    DOT = "."
    DOUBLE_SPACE = SPACE + SPACE
    DOUBLE_DOT = DOT + DOT

    REM_CHARS = ['“', '”', '…', '_', '\u200b', '/', 'ـ', '{', '|', '}', '\xa0', '«', '\u200f', '\u202a',
                 '\u202b', '\u202c', '\u202e', '\u202f', '\u2067', '\u2069', '﴾', '﴿', '»', '’', '•',
                 '\ufeff', '\u200c', '\u200e', 'ۖ', 'ۗ', 'ۚ', 'ٰ', 'ٓ', '‘', '″', '·', '×', 'İ', 'ı', '\x01']

    PUNCTUATIONS = ['!', '؟', '-', '،', '؛', ':', '.']
    PUNCTUATIONS_STR = "".join(sorted(PUNCTUATIONS))

    REP_MAP = {'٠': '0', '١': '1', '٢': '2', '٣': '3', '٦': '6', '٧': '7', '٤': '4', '٥': '5', '٨': '8',
               '٩': '9', '٪': '%', '٫': '،', '—': '-', 'پ': 'ب', 'چ': 'ج', 'ڤ': 'ف', 'ﺋ': 'ئ', 'ﺤ': 'ح', 'ﺧ': 'خ',
               'ﻄ': 'ط', '٬': '،', 'ی': 'ى', 'ﻋ': 'ء', 'ﻖ': 'ق', 'ﻣ': 'م', 'ﻦ': 'ن', 'ﻮ': 'و', 'ﻲ': 'ي', 'ﻼ': 'لا',
               'ٓ': 'ا', '–': '-', 'ﺎ': 'ا', 'ﺑ': 'ب', 'ﺔ': 'ة', 'ﺘ': 'ت', 'ﺮ': 'ر', 'ﺳ': 'س', 'ﺸ': 'ش', 'ﻘ': 'ق',
               'ﻟ': 'ل', 'ﻧ': 'ن', 'ﻬ': 'ه', 'ﻳ': 'ي', 'ﻴ': 'ي', 'ﻵ': 'لآ', 'ﻻ': 'لا', '. . .': '.', '. .': '.',
               '،.': '،', 'ﷺ': SPACE + 'صلى الله عليه وسلم' + SPACE, '©': SPACE + 'copy-right' + SPACE}

    SYMBOLS_TO_RM_PREPR = ["\"", "\'", "“", "”", "«", "»", "(", ")", "<", ">"]
    SYMBOLS_TO_RM_PREPR_RE = re.compile(r'([' + ''.join(SYMBOLS_TO_RM_PREPR) + r'])')
    WS_RE = re.compile(r"\s+")

    TEXT_CASE = "cased"
    LANGUAGE = "ar"
    X01 = '\x01'
    PHONEME_SEPERATOR = ""
    DIACRITIZER_HOST = "77.242.240.151"
    DIACRITIZER_PORT = "5050"


class Formatter:

    def __init__(self) -> None:
        super().__init__()
        self.__normalizer = Normalizer(input_case=Settings.TEXT_CASE, lang=Settings.LANGUAGE)

    def format(self, text) -> str:
        texts = []
        text = self.__normalizer.normalize(text, False, True, True)
        for sent in sentences(text, lang="ar"):
            sentence = sent.text.strip().replace(Settings.DOUBLE_SPACE, Settings.SPACE).strip()
            punct_in_end = False
            for ch in Settings.PUNCTUATIONS:
                sentence = sentence.replace(Settings.SPACE + ch, ch + Settings.SPACE)
                sentence = sentence.replace(ch + ch, ch + Settings.SPACE)
                sentence = sentence.replace(ch + ".", ch + Settings.SPACE)
                sentence = sentence.replace("." + ch, ch + Settings.SPACE)
                sentence = sentence.replace("-" + ch, ch + Settings.SPACE)
                sentence = sentence.replace(ch + "-", ch + Settings.SPACE)
                sentence = sentence.replace(Settings.DOUBLE_SPACE, Settings.SPACE)

                if sentence.startswith(ch):
                    sentence = sentence.replace(ch, Settings.SPACE, 1)
                    sentence = sentence.replace(Settings.DOUBLE_SPACE, Settings.SPACE).strip()

                if sentence.endswith(ch):
                    punct_in_end = True

            if not punct_in_end:
                sentence = sentence + Settings.DOT

            sentence = sentence.replace(Settings.X01, Settings.SPACE)
            sentence = sentence.replace(Settings.DOUBLE_SPACE, Settings.SPACE).strip()
            if sentence and len(sentence) > 0:
                texts.append(sentence)
        return " ".join(texts).strip()


class Cleaner:

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def clean(text: str) -> str:
        text = text.strip("\n").strip("\t").strip()
        text = re.sub(Settings.WS_RE, " ", text).strip()
        text = text.replace("\n", Settings.SPACE).replace("\t", Settings.SPACE)

        for ch in Settings.REM_CHARS:
            text = text.replace(ch, Settings.SPACE)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)

        for ch_k, ch_v in Settings.REP_MAP.items():
            text = text.replace(ch_k, ch_v)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)

        for ch in Settings.PUNCTUATIONS:
            text = text.replace(Settings.SPACE + ch, ch + Settings.SPACE)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)
            text = text.replace(ch, ch + Settings.SPACE)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)
            text = text.strip().replace(ch + ch, ch)
            text = text.strip().replace(Settings.DOUBLE_SPACE, Settings.SPACE)

        for i in range(2):
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)
            text = text.replace(Settings.DOUBLE_DOT, Settings.SPACE)
            text = text.replace(Settings.DOUBLE_SPACE, Settings.SPACE)

        return text


class TextMaker:

    def __init__(self) -> None:
        super().__init__()
        self.__cleaner = Cleaner()
        self.__formatter = Formatter()
        self.diacritizer = MSADiacritizer()

    def make(self, text: str) -> list:
        print(text)
        text = self.__cleaner.clean(text)
        texts = self.__formatter.format(text)
        texts_diac = texts
        try:
            texts_diac = self.diacritizer.diacritize(texts)
        except TypeError:
            try:
                if len(texts.split()) == 1:
                    new_text = texts + " " + texts
                    texts_diac = self.diacritizer.diacritize(new_text).split()[0]
            except Exception as e:
                _log.error(e)
        return texts_diac


class MultilingualNormalizer:
    def __init__(self):
        self.languages = [Language.ENGLISH, Language.ARABIC]
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        self.arb_verbalizer = TextMaker()
        self.normalizer = Normalizer(input_case="cased", lang="en")

    def clean_text(self, text):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251"  # Enclosed characters
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        # Remove multiple spaces
        # text = re.sub(r'\s+', ' ', text)

        # Remove repeating punctuations
        punctuations = r'!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
        text = re.sub(r'([' + re.escape(punctuations) + r'])\1+', r'\1', text)

        # Strip spaces from the start and end
        text = text.strip()

        return text

    def normalize(self, sentence):
        normalized_sentence = ""

        # Segmenting the sentence by language
        for result in self.detector.detect_multiple_languages_of(sentence):
            language_segment = sentence[result.start_index:result.end_index]
            if result.language == Language.ARABIC:
                language_segment = self.arb_verbalizer.make(text=language_segment)
            if result.language == Language.ENGLISH:
                if len(language_segment.split()) > 500:
                    sentence_list = self.normalizer.split_text_into_sentences(language_segment)
                    norm_sentence_list = self.normalizer.normalize_list(sentence_list, False, True, True,
                                                                        batch_size=len(sentence_list), n_jobs=-1)
                    language_segment = " ".join(norm_sentence_list).strip()
                else:
                    language_segment = self.normalizer.normalize(language_segment, False, True, True)

            # Normalize segment
            normalized_segment = language_segment
            normalized_sentence += normalized_segment + " "

        return normalized_sentence.strip()


if __name__ == "__main__":
    normalizer = MultilingualNormalizer()
    sentence = "Hello World from covid-19 125 mm! الانحياز للعلم والديمقراطية والمساواة وحرية المعتقد والتعبير وحق الإنسان في العيش الكريم"
    sentence = normalizer.clean_text(sentence)
    normalized_text = normalizer.normalize(sentence)
    print(normalized_text)
    print(sentence)