import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plot
import librosa
import matplotlib.colors as colors
import os
from sklearn import preprocessing
from IPython import embed
import random
random.seed(12345)


def load_audio(_audio_filename):
    """
    Load audio file

    :param _audio_filename:  
    :return: _y: audio samples
    :return: _fs: sampling rate
    """
    _fs, _y = wav.read(_audio_filename)
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # short int to float
    return _y, _fs


def extract_feature(_audio_filename, nb_mel_bands, nb_frames, nfft):
    # User set parameters
    win_len = nfft
    hop_len = win_len // 2
    window = np.hamming(win_len)
    # nb_mel_bands = 40

    # load audio
    _y, _fs = load_audio(_audio_filename)

    # audio_length = len(_y)
    # nb_frames = int(np.floor((audio_length - win_len) / float(hop_len)))

    # Precompute FFT to mel band conversion matrix
    fft_mel_bands = librosa.filters.mel(_fs, nfft, nb_mel_bands, fmin=0.0).T

    _mbe = np.zeros((nb_frames, nb_mel_bands))

    frame_cnt = 0
    for i in range(nb_frames):
        # framing and windowing
        y_win = _y[i * hop_len:i * hop_len + win_len] * window

        # calculate energy spectral density
        _fft_en = np.abs(fft(y_win)[:1 + nfft // 2]) ** 2

        # calculate mel band energy
        _mbe[frame_cnt, :] = np.dot(_fft_en, fft_mel_bands)

        frame_cnt = frame_cnt + 1
    return _mbe

# -------------------------------------------------------------------
#              Main script starts here
# -------------------------------------------------------------------


window_length = 8  # Window length in samples
nb_mel_bands = 1  # Number of Mel-bands to calculate Mel band energy feature in
nb_frames = 88  # Extracts max_nb_frames frames of features from the audio, and ignores the rest.
#  For example when max_mb_frames = 40, the script extracts features for the first 40 frames of audio
#  Where each frame is of length as specified by win_len variable in extract_feature() function


output_feat_name = 'notentail_vs_entail_{}_{}_{}.npz'.format(nb_frames, nb_mel_bands, window_length)
output_filelist_name = output_feat_name.replace('.npz', '_filenames.npy')
print('output_feat_name: {}\noutput_filelist_name: {}'.format(output_feat_name, output_filelist_name))

# location of data. #TODO: UPDATE ACCORDING TO YOUR SYSTEM PATH
entailment_folder = 'entail'
notentailment_folder = 'notentail'


# Generate training and testing splits
training_ratio = 0.9  # 80% files for training

notentailment_train_files = ''
notentailment_test_files = ''

entailment_train_files = ''
entailment_test_files = ''

# Extract training features
entailment_train_data = np.zeros((nb_frames, nb_mel_bands))
notentaiment_train_data = np.zeros((nb_frames, nb_mel_bands))
#for ind in range(nb_train_files):
entailment_train_data[nb_frames:nb_frames + nb_frames] = extract_feature(
    os.path.join(entailment_folder, entailment_train_files),
    nb_mel_bands,
    nb_frames,
    window_length
    )
notentailment_train_data[nb_frames:nb_frames + nb_frames] = extract_feature(
    os.path.join(notentailment_folder, notentailment_train_files),
    nb_mel_bands,
    nb_frames,
    window_length
    )

# Extract testing features
entailment_test_data = np.zeros((nb_test_files * nb_frames, nb_mel_bands))
notentaiment_test_data = np.zeros((nb_test_files * nb_frames, nb_mel_bands))
for ind in range(nb_test_files):
    music_test_data[nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(entailment_folder, entailment_files),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    speech_test_data[ nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(notentailment_folder, notentailment_files),
        nb_mel_bands,
        nb_frames,
        window_length
    )

# Plotting function to visualize training and testing data before normalization
plot.figure()
plot.subplot2grid((2, 3), (0, 0), colspan=2), plot.imshow(notentailment_train_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=notentailment_train_data.min(), vmax=notentailment_train_data.max()))
plot.title('TRAINING DATA')
plot.xlabel('NOTENTAILMENT - paragraphs')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (0, 2)), plot.plot(np.mean(entailment_train_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean entropy')
plot.subplot2grid((2, 3), (1, 0), colspan=2), plot.imshow(entailment_train_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=entailment_train_data.min(), vmax=entailment_train_data.max()))
plot.xlabel('ENTAILMENT - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (1, 2)), plot.plot(np.mean(entailment_train_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean entropy')

plot.figure()
plot.subplot2grid((2, 3), (0, 0), colspan=2), plot.imshow(notentailment_test_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=notentailment_test_data.min(), vmax=notentailment_test_data.max()))
plot.title('TRAINING DATA')
plot.xlabel('NOTENTAILMENT - paragraphs')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (0, 2)), plot.plot(np.mean(entailment_test_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean entropy')
plot.subplot2grid((2, 3), (1, 0), colspan=2), plot.imshow(entailment_test_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=entailment_test_data.min(), vmax=entailment_test_data.max()))
plot.xlabel('ENTAILMENT - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (1, 2)), plot.plot(np.mean(entailment_test_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean entropy')


# Concatenate speech and music data into training and testing data
train_data = np.concatenate((music_train_data, speech_train_data), 0)
test_data = np.concatenate((music_test_data, speech_test_data), 0)

np.save(output_filelist_name, music_test_files + speech_test_files)

# Labels for training and testing data
train_labels = np.concatenate((np.ones(music_train_data.shape[0]), np.zeros(speech_train_data.shape[0])))
test_labels = np.concatenate((np.ones(music_test_data.shape[0]), np.zeros(speech_test_data.shape[0])))

# Normalize the training data, and scale the testing data using the training data weights
scaler = preprocessing.StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Save labels and the normalized features
np.savez(output_feat_name, train_data, train_labels, test_data, test_labels)
print('output_feat_name: {}\noutput_filelist_name: {}'.format(output_feat_name, output_filelist_name))
plot.show()
