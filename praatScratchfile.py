import os
import math

import parselmouth
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from praatio import tgio
from parselmouth.praat import call, run_file

from sklearn.externals import joblib
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier


class EndingFinder:
    """
    Estimates a range corresponding to the last spoken
    """

    def __init__(self, sound_name):

        self.window_size = 800 // 5
        self.text_grid_path = "TextGrids/"
        self.window_length = 0.07  # window size
        self.syllable_duration = self.window_length * 2  # minimum required audio duration for corresponding window_length

        self.force_minimum_duration = True
        self.cleanup = False

        self.wav_name = sound_name
        self.wav_path = "AudioFiles/" + sound_name + '.wav'
        self.sound = parselmouth.Sound(self.wav_path)

        self.pitch_sampling_factor = 0.005

        self.end_time = self.get_end_of_spoken_time()
        self.last_syllable_time = self.get_last_syllable_time()
        self.f0_frequencies = self.extract_last_syllable_pitch()


    def get_end_of_spoken_time(self):
        sound_data = self.sound.values.T
        number_of_frames = self.sound.get_number_of_frames()

        number_of_windows = number_of_frames // self.window_size

        final_index = self.window_size * number_of_windows
        sound_data = sound_data[:final_index]

        sound_data = np.reshape(sound_data, (-1, self.window_size))
        sound_data = np.power(sound_data, 2)

        data = np.mean(np.abs(sound_data), axis=1)
        data = minmax_scale(data, feature_range=(0, 1), axis=0, copy=True)

        zero_length_array = []
        data[data < 0.1] = 0
        pre = 0
        count = 0
        z_start = 0
        threshold = 30
        end_frame = None
        data[data.size - 1] = 1

        for i in range(data.size):
            if pre == 0 and data[i] == 0:
                count += 1
            elif pre != 0 and data[i] == 0:
                count = 0
                z_start = i
            elif pre == 0 and data[i] != 0:
                if count > threshold and z_start > 0:
                    # print "end point: ", z_start
                    end_frame = z_start
                    # break
                zero_length_array.append((count, z_start))
                count = 0
            pre = data[i]

        zero_length_array.append((data.size - z_start, z_start))

        if not end_frame:
            end_frame = zero_length_array[-1][1]
        for i, (count, frame) in enumerate(zero_length_array):
            if frame == end_frame and i < len(zero_length_array) - 1:
                next_count = zero_length_array[i + 1][0]
                end_frame = end_frame + int(math.floor(next_count * 0.5))
                break

        frame = (end_frame) * self.window_size
        print(frame)

        return self.sound.frame_number_to_time(frame)


    def get_last_syllable_time(self):
        """
        Parses a TextGrid object, returning the time of the beginning of the nucleus of the last syllable
        """
        text_grid_name = self.text_grid_path + self.wav_name + '.syllables.TextGrid'
        tg = tgio.openTextgrid(text_grid_name)
        syllableList = tg.tierDict["syllables"].entryList[::-1]  # Get all intervals, beginning from the end

        if len(syllableList) < 1:
            raise AssertionError('Was unable to find any speech in sound-file')

        if self.cleanup:
            os.remove(text_grid_name)

        for i in range(0, len(syllableList)):
            time_stamp = syllableList[i].time
            if time_stamp < self.end_time:
                duration = self.end_time - time_stamp
                if duration > self.syllable_duration or not self.force_minimum_duration:
                    return time_stamp

        # return the 'first' marked syllable in time
        return syllableList[-1].time


    def draw_pitch(self):
        """
        :param pitch: parselmouth.Sound(.wav path).to_pitch() object
        """
        plt.figure()
        plt.twinx()

        # Extract selected pitch contour, and
        # replace unvoiced samples by NaN to not plot
        last_segment = self.sound.extract_part(self.last_syllable_time, self.end_time)
        last_pitches = last_segment.to_pitch(self.pitch_sampling_factor)

        pitch_values = np.copy(self.get_f0_frequency())
        pitch_values[pitch_values==0] = np.nan

        plt.plot(last_pitches.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(last_pitches.xs(), pitch_values, 'o', markersize=2)
        plt.grid(False)
        plt.ylim(0, last_pitches.ceiling)
        plt.ylabel("fundamental frequency [Hz]")

        plt.xlim([last_segment.xmin, last_segment.xmax])
        plt.show()


    def extract_last_syllable_pitch(self):
        """

        :param wav_name:
        :return:
        """
        print("Extracting syllable intervals from '{}'...".format(self.wav_path))

        run_file('single_syllable_nuclei.praat', -25, 2, 0.3, True, self.text_grid_path, self.wav_path)

        # get beginning of last syllable and end of file time

        print('Last syllable duration:', self.last_syllable_time ,'-', self.end_time)
        last_segment = self.sound.extract_part(self.last_syllable_time, self.end_time)

        # the bigger the number, the larger the time-step for pitch-sampling (0 is default).
        final_segment_pitch = last_segment.to_pitch(self.pitch_sampling_factor)

        f0_frequencies = final_segment_pitch.selected_array['frequency']
        # print('-------------------------------------------')
        # print(f0_frequencies)
        # print('-------------------------------------------')

        return f0_frequencies


    def get_f0_frequency(self):
        return self.f0_frequencies



class F0ApproximationCurveExtractor:
    """最終モーラのF0波形を3次の最小二乗法で近似 Approximate final mora's F0 waveform by third-order least squares method"""

    def __init__(self):
        pass

    def extract(self, f0List):

        print("f0 Approximation!!!")

        ### 前処理．F0を対数化し，正規化する Preprocessing: log and normalize F0
        def normalize(x):
            # """平均0，分散1に正規化する．np.nanは無視する Normalize to mean 0, variance 1. Ignore np.nan"""
            xmin = np.nanmin(x)
            xmax = np.nanmax(x)
            xcenter = (xmax + xmin) / 2.0
            xfactor = (xmax - xmin) / 2.0
            return (x - xcenter) / xfactor

        # F0リストをnp.array化し，0をnp.nanにする Np.array 0 list and set 0 to np.Nan
        f0_array = np.array(f0List)
        f0_array[f0_array < 1e-10] = np.nan

        # 対数化する（np.nanは無視される） Convert to logarithm, (np.nan is ignored)
        log_f0_array = np.log10(f0_array)

        # 正規化する（発話全体の対数F0平均を0，分散を1にする）Normalize (set LogF0 average of whole utterance to 0, variance to 1)
        norm_log_f0_array = normalize(log_f0_array)

        norm_log_f0_last_mora_array = norm_log_f0_array

        while np.isnan(norm_log_f0_last_mora_array)[0] == True:
            norm_log_f0_last_mora_array = np.delete(norm_log_f0_last_mora_array,0)

        while np.isnan(norm_log_f0_last_mora_array)[-1] == True:
            norm_log_f0_last_mora_array = np.delete(norm_log_f0_last_mora_array,-1)


        ### 近似のためのX軸の値の準備 Preparing X-axis values for approximation
        x_array = np.array(range(norm_log_f0_last_mora_array.size))

        # F0が np.nan の部分を取り除く F0 removes the part of np.nan
        flag = np.isnan(norm_log_f0_last_mora_array) == False
        trimmed_norm_log_f0_last_mora_array = norm_log_f0_last_mora_array[flag]
        trimmed_x_array = x_array[flag]

        ### 3次関数で近似 Approximate by cubic function
        def func(x, a, b, c, d):
            return a + b * x + c * x * x + d * x * x * x

        param, cov = opt.curve_fit(func,
                                   trimmed_x_array, trimmed_norm_log_f0_last_mora_array)

        ### 近似した3次曲線の値 Approximate cubic curve value
        approx_curve = func(x_array, *param)
        approx_curve_zero_base = approx_curve - approx_curve[0]

        ### 代表点を11点取り出す Take 11 key points
        x_feature = np.linspace(0.0, norm_log_f0_last_mora_array.size, 11)
        y_feature = func(x_feature, *param)

        ### 0番目の点との差にする Difference with the 0th point
        y_feature_zero_base = y_feature - y_feature[0]

        return y_feature_zero_base


class QuestionClassifier:
    """認識器 Classifier: probability a sound clip is a question"""

    def __init__(self):
        self.classifier = None
        pass

    def train(self, train_csv_file, label_csv_file):
        train_data = np.loadtxt(train_csv_file, delimiter=',')
        label_data = np.loadtxt(label_csv_file, delimiter=',')
        print ("train classifier!!")# Train classifier

        train_data = train_data[:, :11]
        train_data[np.isnan(train_data)] = np.finfo(np.float32).min

        classifier = RandomForestClassifier()
        classifier.fit(train_data, label_data)

        self.classifier = classifier


    def loadmodel(self, model_name):
        classifier = joblib.load(model_name)

        self.classifier = classifier

    def predict(self, curve_extractor):
        print("classifier predict!!")
        feature = np.hstack([curve_extractor])
        feature[np.isnan(feature)] = np.finfo(np.float32).min

        result = self.classifier.predict(feature.reshape(1, -1))

        self.result = result


    def probability(self, curve_extractor):
        print("classifier predict!!")
        feature = np.hstack([curve_extractor])
        feature[np.isnan(feature)] = np.finfo(np.float32).min

        result = self.classifier.predict_proba(feature.reshape(1, -1))
        self.result = result[0,1]

    def get_result(self):
        return self.result


ef = EndingFinder(input('Enter name (without .wav) of .wav file: '))
f0List = ef.get_f0_frequency()

classifier = QuestionClassifier()
classifier.train("data/datas.csv", "data/labels.csv")
curve_extractor = F0ApproximationCurveExtractor()
classifier.probability(curve_extractor.extract(f0List))
print(classifier.get_result())




