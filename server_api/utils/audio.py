from pydub import AudioSegment


def resample_and_save_as_mp3(input_array, input_sample_rate, output_mp3_path, target_sample_rate=44100):
    audio_segment = AudioSegment(
        input_array.tobytes(),
        sample_width=input_array.dtype.itemsize,
        frame_rate=input_sample_rate,
        channels=1
    )

    audio_segment = audio_segment.set_frame_rate(target_sample_rate)
    audio_segment.export(output_mp3_path, format="mp3")
