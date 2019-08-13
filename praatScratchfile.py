import parselmouth
from parselmouth.praat import call, run_file
from praatio import tgio
import numpy as np
import scipy.optimize as opt
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import minmax_scale

audio_path = "AudioFiles/"
text_grid_path = "TextGrids/"
window_length = 0.7
syllable_duration = 0.15 # > 2 * window_length

minimum_syllable_duration = True

def get_end_time_at_silence(sound):
    window_size = 800 // 5   # 160

    sound_data = sound.values.T
    number_of_frames = sound.get_number_of_frames()

    number_of_windows = number_of_frames // window_size

    final_index = window_size * number_of_windows
    sound_data = sound_data[:final_index]
    sound_data = np.reshape(sound_data, (-1, window_size))
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
    # print zero_length_array
    if not end_frame:
        end_frame = zero_length_array[-1][1]
    for i, (count, frame) in enumerate(zero_length_array):
        if frame == end_frame and i < len(zero_length_array) - 1:
            next_count = zero_length_array[i + 1][0]
            end_frame = end_frame + int(math.floor(next_count * 0.5))
            break

    frame = (end_frame) * window_size
    print(frame)

    return sound.frame_number_to_time(frame)


def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan

    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

# pitch = snd.to_pitch()

'''
# If desired, pre-emphasize the sound fragment before calculating the spectrogram
pre_emphasized_snd = snd.copy()
pre_emphasized_snd.pre_emphasize()
spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
plt.figure()
plt.twinx()
draw_pitch(pitch)
plt.xlim([snd.xmin, snd.xmax])
plt.show() # or plt.savefig("spectrogram_0.03.pdf")
'''

def get_last_syllable_time(filename, end_time):
    """
    Parses a TextGrid object, returning the time of the beginning of the nucleus of the last syllable
    :param filename: Name of .wav (excluding the .wav extension) file that has a TextGrid object
    :return:
    """

    tg = tgio.openTextgrid(text_grid_path + filename + '.syllables.TextGrid')
    syllableList = tg.tierDict["syllables"].entryList  # Get all intervals


    for i in range(len(syllableList) - 1, -1, -1):
        temp_start = syllableList[i].time
        if temp_start < end_time:
            duration = end_time - temp_start
            if duration > syllable_duration or not minimum_syllable_duration:
                return syllableList[i].time

    return syllableList[0].time
    # raise AssertionError('The end-time for silence is earlier than the first-known syllable time')


def extract_last_syllable_pitch():
    """

    :return:
    """

    wav_name = input('Name of audio file: ')
    file_name, _ = wav_name.split('.')
    wav_path = audio_path + wav_name

    # TODO: take an excerpt of this one first to get the valid/cut off file
    sound = parselmouth.Sound(wav_path)

    print("Extracting syllable intervals from '{}'...".format(wav_path))

    run_file('single_syllable_nuclei.praat', -25, 2, 0.3, True, text_grid_path, wav_path)

    # get beginning of last syllable and end of file time
    end_time = get_end_time_at_silence(sound)
    print("!!! End-time: ", end_time)
    # end_time = sound.get_total_duration()
    last_syllable_time = get_last_syllable_time(file_name, end_time)

    print('Last syllable duration:', last_syllable_time ,'-', end_time)
    last_segment = sound.extract_part(last_syllable_time, end_time)

    # the bigger the number, the larger the time-step for pitch-sampling (0 is default).
    final_segment_pitch = last_segment.to_pitch(0.005)

    pitch_values = final_segment_pitch.selected_array['frequency']
    # pitch_values[pitch_values==0] = np.nan

    print('-------------------------------------------')
    print(pitch_values)
    print(pitch_values.size)
    print('-------------------------------------------')

    if minimum_syllable_duration:
        spectrogram = last_segment.to_spectrogram(window_length=0.007, maximum_frequency=8000)
        plt.figure()
        plt.twinx()
        draw_pitch(final_segment_pitch)
        plt.xlim([last_segment.xmin, last_segment.xmax])
        plt.show()

    return pitch_values




class F0ApproximationCurveExtractor:
    """最終モーラのF0波形を3次の最小二乗法で近似 Approximate final mora's F0 waveform by third-order least squares method"""

    def __init__(self):
        pass

    def extract(self, f0List):

        print("f0 Approximation!!!")

        ### 前処理．F0を対数化し，正規化する Preprocessing: log and normalize F0
        def normalize(x):
            # """平均0，分散1に正規化する．np.nanは無視する Normalize to mean 0, variance 1. Ignore np.nan"""
            # mu  = np.nanmean(x)
            # std = np.nanstd(x)
            # return (x - mu) / std
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

        # ### 最終モーラのF0を抽出する Extract the final mora's F0
        # # 最終モーラのフレーム，終了フレームを取得 Get the frames pertaining to the last mora
        # endFrame = mora.moraFrameList[-2] # 最終モーラの終了フレーム End frame of last mora
        # if len(mora.moraFrameList) > 1:
        # 	startFrame = mora.moraFrameList[-3] # 最終モーラの開始 Start frame of last mora
        # else:
        # 	startFrame = 0

        # TODO: norm_log_f0_last_mora_array == what was found in code above already
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

        # print trimmed_x_array
        # print trimmed_norm_log_f0_last_mora_array

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

        print(y_feature_zero_base)

        return y_feature_zero_base

        # ### 直前の値との差にする（要素数10のベクトルになる）Make a difference with the previous value (becomes a vector w/ 10 elements)
        # y_feature_diff = y_feature[1:] - y_feature[:-1]
        #
        # ### 結果を公開用にインスタンス変数にする Make the result an instance variable for publishing
        # self.normLogF0LastMoraArray = norm_log_f0_last_mora_array
        # self.trimmedXArray = trimmed_x_array
        # self.trimmedNormLogF0LastMoraArray = trimmed_norm_log_f0_last_mora_array
        # self.approxCurve = approx_curve
        # self.approxCurveZeroBase = approx_curve_zero_base
        # self.feature = y_feature
        # self.featureZeroBase = y_feature_zero_base
        # self.featureDiff = y_feature_diff
        #
        # self.f0_wave_11_point = y_feature_zero_base

class Classifier:
    """認識器 Recognizer"""

    def __init__(self):
        self.classifier = None
        pass

    def train(self, train_csv_file, label_csv_file):
        train_data = np.loadtxt(train_csv_file, delimiter=',')
        label_data = np.loadtxt(label_csv_file, delimiter=',')
        print ("classifier train!!")# Train classifier

        train_data = train_data[:, :11]
        train_data[np.isnan(train_data)] = np.finfo(np.float32).min

        classifier = RandomForestClassifier()
        classifier.fit(train_data, label_data)

        # print "\n\n"
        # print "RandomForestClassifier.feature_importances_"
        # print classifier.feature_importances_
        # print "\n\n"

        self.classifier = classifier


    def loadmodel(self, model_name):
        classifier = joblib.load(model_name)

        self.classifier = classifier

    def predict(self, curve_extractor):
        print("classifier predict!!")
        feature = np.hstack([
            curve_extractor])
        feature[np.isnan(feature)] = np.finfo(np.float32).min

        result = self.classifier.predict(feature.reshape(1, -1))

        self.result = result


    def probability(self, curve_extractor):
        print("classifier predict!!")
        feature = np.hstack([
            curve_extractor])
        feature[np.isnan(feature)] = np.finfo(np.float32).min

        result = self.classifier.predict_proba(feature.reshape(1, -1))
        self.result = result[0,1]

f0List = extract_last_syllable_pitch()

classifier = Classifier()
classifier.train("data/datas.csv", "data/labels.csv")
curve_extractor = F0ApproximationCurveExtractor()
classifier.probability(curve_extractor.extract(f0List))
print(classifier.result)




