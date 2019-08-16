# from pyAudioAnalysis import audioBasicIO as aIO
# from pyAudioAnalysis import audioSegmentation as aS

# [Fs, x] = aIO.readAudioFile("WasedaWavs/0a7f7fb7d0993fe5ba1891da8c58d79f.wav")
# segments = aS.silenceRemoval(x, Fs, 0.020, 0.020, smoothWindow = 1.0, weight = 0.3, plot = True)

###########################################

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
