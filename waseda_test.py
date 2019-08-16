from question_classifier import QuestionClassifier, EndingFinder, F0ApproximationCurveExtractor

import os

import parselmouth
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

import config

# Generate and save a pdf showing the area where
make_pdf = config.setting_bool['make_pdf']

out_pdf = './vad_3_sound.pdf'
pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)

def classify_sound(ending_finder, classifier):
    """
    Classifies a .wav file located in the "AudioFiles/" folder with the name sound_name (no .wav extension)
    :param ef: EndingFinder object loaded with audio's name
    :return: probability that the .wav file is a question
    """
    f0List = ending_finder.get_f0_frequency()

    curve_extractor = F0ApproximationCurveExtractor()
    classifier.probability(curve_extractor.extract(f0List))

    return classifier.get_result()


def is_question(waseda_classifier_result, is_classifier_correct):
    is_question = None
    if waseda_classifier_result == 'true':
        if is_classifier_correct == 'correct':
            is_question = True
        else:
            is_question = False

    else:
        if is_classifier_correct == 'correct':
            is_question = False
        else:
            is_question = True

    return is_question


def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values < 1e-10] = np.nan
    # pitch_values[pitch_values > 300] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")


def draw_figure(snd, start_time, end_time, title, text):

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.title(title)
    plt.axvline(x=start_time, color='y')
    plt.axvline(x=end_time, color='r')

    plt.subplot(122)

    pitch = snd.to_pitch()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()

    draw_pitch(pitch)
    plt.title(title)
    plt.xlim([snd.xmin, snd.xmax])
    plt.axvline(x=start_time, color='g', linewidth=.5)
    plt.axvline(x=end_time, color='r', linewidth=.5)

    plt.text(0, 0, text)
    pdf.savefig(fig)
    # plt.show() # or plt.savefig("spectrogram_0.03.pdf")




if __name__ == '__main__':

    training_data = config.setting_paths['default_data_path']
    labels = config.setting_paths['default_label_path']
    test_data = config.setting_paths['default_test_path']

    classifier = QuestionClassifier()
    classifier.train(training_data, labels)

    entry_array = []
    with open(test_data, 'r') as an_file:
        for line in an_file:
            entry = line.split(',')
            entry_array.append(entry)

    is_question_and_predict_is_question = 0
    is_question_and_predict_not_question = 0

    not_question_and_predict_is_question = 0
    not_question_and_predict_not_question = 0

    total_number = 0
    error_count = 0

    for entry in entry_array:
        file_name = entry[0]
        waseda_classifier_result = entry[4][:-1]  # \n is in this field
        is_classifier_correct = entry[1]

        if is_classifier_correct == 'cancel':
            continue

        total_number += 1

        is_this_sentence_question = is_question(waseda_classifier_result, is_classifier_correct)

        ending_finder = EndingFinder(file_name, config.setting_paths['default_audio_path'])

        question_probability = classify_sound(ending_finder, classifier)

        if is_this_sentence_question:
            if question_probability >= 0.5:
                is_question_and_predict_is_question += 1
            else:
                is_question_and_predict_not_question += 1

        else:
            if question_probability >= 0.5:
                not_question_and_predict_is_question += 1
            else:
                not_question_and_predict_not_question += 1

        if make_pdf:
            draw_figure(
                ending_finder.get_sound(),
                ending_finder.get_syllable_start(),
                ending_finder.get_syllable_end(),
                str(total_number) + ': ' + file_name,
                str(is_this_sentence_question) + ': ' + str(question_probability))

    pdf.close()

    if not make_pdf:
        os.remove('./vad_3_sound.pdf')


    print('total number: ', total_number)
    print('----------------------------------------------------------------------------')
    print('QUESTION: Yes       PREDICTION: CORRECT ', is_question_and_predict_is_question)
    print('QUESTION: Yes       PREDICTION: INCORRECT ', is_question_and_predict_not_question)
    print('----------------------------------------------------------------------------')
    print('QUESTION: No        PREDICTION: CORRECT ', not_question_and_predict_not_question)
    print('QUESTION: No        PREDICTION: INCORRECT ', not_question_and_predict_is_question)
    print('----------------------------------------------------------------------------')
    print('PERCENT CORRECT, OF QUESTIONS: ', is_question_and_predict_is_question * 1.0 / (is_question_and_predict_is_question + is_question_and_predict_not_question))
    print('PERCENT INCORRECT, OF QUESTIONS: ', is_question_and_predict_not_question * 1.0 / (is_question_and_predict_is_question + is_question_and_predict_not_question))
    print('PERCENT CORRECT, OF NOT QUESTIONS: ', not_question_and_predict_not_question * 1.0 / (not_question_and_predict_is_question + not_question_and_predict_not_question))
    print('PERCENT INCORRECT, OF NOT QUESTIONS: ', not_question_and_predict_is_question * 1.0 / (not_question_and_predict_is_question + not_question_and_predict_not_question))
    print('----------------------------------------------------------------------------')
    print('PERCENT CORRECT: ', (is_question_and_predict_is_question + not_question_and_predict_not_question) * 1.0 / total_number)
    print('PERCENT INCORRECT: ', (is_question_and_predict_not_question + not_question_and_predict_is_question) * 1.0 / total_number)
