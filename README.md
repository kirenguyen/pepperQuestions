# pepperQuestions

Question determination prototype for Pepper.


### Python files:

- `question_classifier.py`
- `config.py`
- `waseda_test.py`
- `training_data_generator.py`
- `web_rtc_vad.py`
- `audio_splitting.py`

##### `question_classifier.py`

File that contains three major classes: `QuestionClassifier`, `F0ApproximationCurveExtractor`, and `EndingFinder` that contains the logic for extracting the necessary features for the RandomForestClassifier.

##### `config.py`

Settings that directly relate to the functionality of `question_classifier.py` and `waseda_test.py`. See the file's comments and description of other files for specific usage.

​		Remember to check the configurations for the correct paths to your audio files!


##### `training_data_generator.py`:

Generator currently creates basic training data based on annotated files from Waseda to reprocess the audio files, used mostly for testing. The new CSV data is generated into `new_training_data/datas.csv` and `new_trainng_data/labels.csv` using the respective files in that folder.


##### `waseda_test.py`

File that tests the correctness of the current feature extraction and trained model using Waseda's annotated data. To see a generated PDF of the area where `question_classifier.py` pulled features from, set the `make_pdf` boolean to `True` in `config.py`. 

The model will be trained using the paths specified in `config.py`, as long as they are annotated and in the same format as the original Waseda CSVs: ['data/datas.csv', 'data/labels.csv', 'data/annotation_no_cancel.csv')

##### `web_rtc_vad.py`

Slightly modified 

##### `audio_splitting.py` 

Scratch file testing out the undocumented functionality of a library that attempts to find intervals of voiced speech, pyAudioAnalysis and pydub.

### Main data directories:

- data/

- new_training_data/

- WasedaWavs/

  

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

- current training data ('data/datas.csv') was not used creating the same feature-extraction code.
- Japanese vs English discrepancies
- Noise clean-up results in the end of sentences, where the voice tends to get less loud, to lose pitch points, resulting in poorer accuracy with the trained model.
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
- consider using a more dedicated voice activity detection library, like: https://github.com/wiseman/py-webrtcvad
    + This is currently the most-up-to-date VAD library, used in most modern browsers and is free for use.
- Some quickly written example usages of a few other libraries that attempt to select speech can be found in `audio_splitting.py`
- I suspect feature extraction from the last estimated syllable is far too naive, especially in noisy rooms or family-rooms. Speaker diarization and audio segmentation will be a key pre-processing step.
    + [PyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) may have some help with this