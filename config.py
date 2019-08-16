'''
Values used to determine fundamental frequency in question_classifier.py, must be real, non-negative numbers.

Window_length: A value of 0.07 has resulted in the best accuracy so-far
Pitch_sampling_factor: the number of the measurement interval (frame duration), in seconds.
    If you supply 0, Praat will use a time step of 0.75 / (pitch floor), e.g. 0.01 seconds if the pitch floor is 75 Hz
Vad_intensity: An integer in (0, 1, 2, 3) where 0 is the weakest level of voice-less sound removal and 3 is the strongest
'''
setting_val = {
    'window_length' : 0.07,             # > 0
    'pitch_sampling_factor' : 0.001,    # >= 0

    'vad_intensty' : 3
}


'''
Boolean values that are used in question_classifier.py and waseda_test.py

clean_audio: Removes noise through spectral subtraction via single_preprocess_audio.praat if True
normalize_audio: Normalizes audio in pre-processing via single_preprocess_audio.praat if True
force_minimum_duration: Ensures that the sample taken from the audio files is 2 * window_length, recommended to keep True
cleanup: Delete intermediate files created by running Praat scripts if True, saves them to temp_data if False
make_pdf: saves a PDF of the segments from waseda_test.py
'''

setting_bool = {
    'clean_audio': False,
    'normalize_audio' : False,  # if clean_audio is False, audio will not be normalized

    'force_minimum_duration' : True,
    'cleanup': True,    # if False, saves intermediate files to temp_data

    'make_pdf' : False,
}


'''
Default paths used for training and testing in waseda_test.py.

Default paths must be to files of the same format as the ones located in the data/ subdirectory.
'''
setting_paths = {
    'default_data_path' : 'data/datas.csv',
    'default_label_path' : 'data/labels.csv',
    'default_test_path' : 'data/annotation_edited.csv',

    'default_audio_path' : 'waseda_wavs/',

    'default_vad_path' : 'vad_filtered_audio/', # path to save audio files for web_rtc_vad.py
}