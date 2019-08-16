# pepperQuestions

Question determination prototype for Pepper.


### Python files:

- `question_classifier.py`

- `config.py`

- `waseda_test.py`

- `training_data_generator.py`

- `web_rtc_vad.py`

- `audio_splitting.py`

  

#### ` question_classifier.py`

File that contains three major classes: `QuestionClassifier`, `F0ApproximationCurveExtractor`, and `EndingFinder` that contains the logic for extracting the necessary features for the RandomForestClassifier.

#### `config.py`

Settings that directly relate to the functionality of `question_classifier.py`, `waseda_test.py`, and `web_rtc_vad.py`. See the file's comments and description of other files for specific usage.

​Remember to check the configurations for the correct paths to your audio files!


#### `training_data_generator.py`:

Generator currently creates basic training data based on annotated files from Waseda to reprocess the audio files, used mostly for testing. The new CSV data is generated into `new_training_data/datas.csv` and `new_trainng_data/labels.csv` using the respective files in that folder.


#### `waseda_test.py`

File that tests the correctness of the current feature extraction and trained model using Waseda's annotated data. To see a generated PDF of the area where `question_classifier.py` pulled features from, set the `make_pdf` boolean to `True` in `config.py`. 

The model will be trained using the paths specified in `config.py`, as long as they are annotated and in the same format as the original Waseda CSVs: [`data/datas.csv`, `data/labels.csv`, `data/annotation_no_cancel.csv`]

#### `web_rtc_vad.py`

Modified sample Web RTC script that takes a .wav file, splits it into voiced chunks, then stitches the .wav file back together. At the moment, it currently results in poorer accuracy due to `question_classifier.py`'s methodology, but may be a good alternative for future iterations.

When running the file, you will be prompted to give a path to an audio file.`web_rtc_vad.py` will take every file that ends in `.wav` and run it through WebRTC's VAD.

In `config.py`, you can choose how much WebRTC tries to filter out the noise/silence by changing `setting_vals['vad_intensity']` from 0 - 3. In `new_training_data`, you can see the difference between how the code works for intensity 2 and 3 by looking at the pre-loaded pdfs (`vad_2_sound.pdf` and `vad_3_sound.pdf`).

#### `audio_splitting.py` 

Scratch file testing out the undocumented functionality of a library that attempts to find intervals of voiced speech, pyAudioAnalysis and pydub similar to WebRTC. The library has more options to personally change how voice is filtered, but is more difficult to approach than Web RTC's simple 4 settings (0-3).

### Main data directories:

- `data/`
    + Contains several CSV files as a result of errors caused by pitch-less splices of .wav files. 
- `new_training_data/` 
    + Contains new training data and some PDFs resulting from running `waseda_test.py` on the same set of .wav files processed in different ways.
        + modified_sound.pdf - run through Praat noise removal
        + original_sound.pdf - original .wav used
        + vad_2_sound.pdf - run through level 2 WebRTC VAD
        + vad_3_sound.pdf - run through level 3 WebRTC VAD
- `WasedaWavs/`
    + The audio files associated with the CSV files in `data/`. Note the folder `noise_removed` contains all of the audio files in the outer directory, but run through the Praat noise removal script.
  

#### Others 

- tempData/ : used as a temporary directory for intermediate files produced from running Praat scripts
- praat/ : contains the praat scripts used in `question_classifier.py`
  - Check the praat scripts for more information about the parameters used in `run_file` calls, but it would be best to leave them as the default ones.


## Example Usage:


Simple script to determine the probability that an audio file is a question.

```python
from question_classifier import QuestionClassifier, EndingFinder, F0ApproximationCurveExtractor


file_name = input('Enter name (without .wav) of sound file: ')
audio_path = "AudioFiles/"  # path to sound directory
ef = EndingFinder(file_name, audio_path)
f0List = ef.get_f0_frequency()  # list of f0 frequencies of ~last syllable

classifier = QuestionClassifier()
# train the classifier 
classifier.train("data/datas.csv", "data/labels.csv")   

curve_extractor = F0ApproximationCurveExtractor()       
classifier.probability(curve_extractor.extract(f0List))

print("PROBABILITY THAT IT IS A QUESTION: " , classifier.get_result())
```



## Problems:

- Current training data (`data/datas.csv`) was not used creating the same feature-extraction code.
- Japanese vs English discrepancies
- Noise clean-up results in the end of sentences, where the voice tends to get less loud, to lose pitch points, resulting in poorer accuracy with the trained model.
    + `web_rtc_vad.py` does decent clean-up, but the method of finding the last syllable/mo-ra needs to be changed to accommodate for the slicing/stitching technique.
- Praat is a finicky research tool, not one meant for production. Alternative methods should be sought to find timing of last mo-ra.
- Major faults with finding the last mora (モーラ); right now, it attempts to fnd the last part of the spoken-speech by using a Syllable Nuclei Script.
    + The method of finding the end of the spoken-segment of the file is inaccurate
    + Japanese vs English discrepancies
    + The code iterates backwards through the starting points of the syllables until it finds something at least ~0.15s away from the end of the spoken-segment; this leads to segments of the file that may result in no pitches to train or predict on
+ There are several varying factors that change whether enough non-NaN features are extracted
    + pitch sampling rate
    + end_frame_factor
    + minimum syllable duration



## Suggestions for next steps:
- At the moment, audio files are pre-processed with standard spectral subtraction via a Praat script for noise cancellation; consider something more heavy-duty for noisier households
    - Noise cancellation
- Consider using a more dedicated voice activity detection library, like: https://github.com/wiseman/py-webrtcvad
    + This is currently the most-up-to-date VAD library, used in most modern browsers and is free for use.
    + `web_rtc_vad.py` was written to be a sort of starting-point; you can see the pdfs to see how it currently performs.
- Some quickly written example usages of a few other libraries that attempt to select speech can be found in `audio_splitting.py`
- I suspect feature extraction from the last estimated syllable is far too naive, especially in noisy rooms or family-rooms. Speaker diarization and audio segmentation will be a key pre-processing step.
    + [PyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) may have some help with this