from praatScratchfile import QuestionClassifier, EndingFinder, F0ApproximationCurveExtractor

def classify_sound(classifier, sound_name):
    """
    Classifies a .wav file located in the "AudioFiles/" folder with the name sound_name (no .wav extension)
    :param sound_name: name of .wav file without the .wav extension
    :return: probability that the .wav file is a question
    """
    ef = EndingFinder(sound_name, "WasedaWavs/")
    f0List = ef.get_f0_frequency()

    # classifier = QuestionClassifier()
    # classifier.train("data/datas.csv", "data/labels.csv")

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

if __name__ == '__main__':

    classifier = QuestionClassifier()
    classifier.train("data/datas.csv", "data/labels.csv")

    entry_array = []
    with open('data/annotation_edited.csv', 'r') as an_file:
        for line in an_file:
            entry = line.split(',')
            entry_array.append(entry)

    # print(entry_array)

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

        question_probablity = classify_sound(classifier, file_name)

        if is_this_sentence_question:
            if question_probablity >= 0.5:
                is_question_and_predict_is_question += 1
            else:
                is_question_and_predict_not_question += 1

        else:
            if question_probablity >= 0.5:
                not_question_and_predict_is_question += 1
            else:
                not_question_and_predict_not_question += 1



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
