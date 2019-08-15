from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent, detect_silence

sound_file = AudioSegment.from_wav("NoiselessAudioFiles/nande_q.wav")
audio_chunks = split_on_silence(sound_file,
    # must be silent for at least .15s
    min_silence_len=150,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-25
)

nonsilent = detect_nonsilent(sound_file,
    # must be silent for at least .15s
    min_silence_len = 150,

    # consider it silent if quieter than -16 dBFS
    silence_thresh = -25
    )

print(nonsilent)

silent = detect_silence(sound_file,
    # must be silent for at least .15s
    min_silence_len = 150,

    # consider it silent if quieter than -16 dBFS
    silence_thresh = -25
    )

print(silent)

# for i, chunk in enumerate(audio_chunks):
#     out_file = "splitAudio/chunk{0}.wav".format(i)
#     print("exporting", out_file)
#     chunk.export(out_file, format="wav")
