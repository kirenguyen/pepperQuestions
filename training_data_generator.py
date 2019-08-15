import csv

from question_classifier import QuestionClassifier, EndingFinder, F0ApproximationCurveExtractor

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


    #optional: clearing the training/label csv files
    f = open("new_training_data/data.csv", "w")
    f.truncate()
    f.close()

    f = open("new_training_data/labels.csv", "w")
    f.truncate()
    f.close()


    entry_array = []
    with open('new_training_data/new_training_files.csv', 'r') as an_file:
        for line in an_file:
            entry = line.split(',')
            entry_array.append(entry)

    features_list = []
    label_list = []
    for entry in entry_array:
        file_name = entry[0]  # file_name without the .wav extension
        waseda_classifier_result = entry[4][:-1]  # \n is in this field
        is_classifier_correct = entry[1]

        ef = EndingFinder(file_name, "WasedaWavs/")
        f0List = ef.get_f0_frequency()

        curve_extractor = F0ApproximationCurveExtractor()
        eleven_points = curve_extractor.extract(f0List)
        features_list.append(eleven_points)

        is_question_audio = is_question(waseda_classifier_result, is_classifier_correct)

        if is_question_audio:
            label_list.append(1.000000000000000000e+00)
        else:
            label_list.append(0.000000000000000000e+00)

    with open('new_training_data/data.csv', mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for list in features_list:
            data_writer.writerow(list)

    with open('new_training_data/labels.csv', mode='w') as label_file:
        label_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for value in label_list:
            label_writer.writerow([value])