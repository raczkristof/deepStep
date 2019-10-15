import numpy as np
from scipy.io import wavfile
from scipy import signal
import noisereduce as nr # https://timsainburg.com/noise-reduction-python.html?fbclid=IwAR33W89I0YfA157qE1OkUFqfwXZ5vE2gVePZ-k_SmMZPaJEzu2hfdbuppE4
import glob
import os
import re

# http://stackoverflow.com/questions/39032325/ddg#39032946
# Butterworth highpass filter
def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=2):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Butterworth notch filter
def butter_notch(cutoff, width, fs, order = 2):
    nyq = 0.5 * fs
    normal_cutoff_beg = (cutoff - width / 2) / nyq
    normal_cutoff_end = (cutoff + width / 2) / nyq
    b, a = signal.butter(order, [normal_cutoff_beg, normal_cutoff_end], btype='bandstop', analog=False)
    return b, a

def butter_notch_filter(data, cutoff, width, fs, order = 2):
    b, a = butter_notch(cutoff, width, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Get the data containing a sample of the noise
fs, noise = wavfile.read('data_raw/noise.wav')

# Raw recordings are in stereo:
# left channel is recorded with a Shure SM-58 dynamic microphone, 
# right channel is recorded with an AudioTechnica - AT2020 condenser microphone
# I have found that if I use the noise recording as it is, I get artifacts in the denoised sound,
# so I apply a highpass filter with a cutoff freq of 1 kHz
noise_profile_shure = butter_highpass_filter(noise[:,0], 1000, fs)
noise_profile_at = butter_highpass_filter(noise[:,1], 1000, fs)

# Find all the raw recordings, and remove the noise recording from the list
files = [f for f in glob.glob("data_raw/*.wav")]
files.remove('data_raw/noise.wav')

# Loop through all the raw files
n = len(files)
for i in range(0, n):

    # Read the raw recording
    fs, data = wavfile.read(files[i])

    # Split based on the microphone
    data_shure = data[:,0]
    data_at = data[:,1]

    # for plotting with actual time instead of sample number
    # t = np.arange(0, len(data_at)/fs, 1/fs)

    # For filtering the recordings, a noise reduction is first performed using the noise samples,
    # then a highpass filter is applied with a cutoff freq of 100 Hz,
    # lastly, a notch filter is applied between 43 and 47 Hz to get rid of an annoying humm in that region

    # Perform the filtering on the recordings of the Shure microphone
    filt_data_at = butter_notch_filter(butter_highpass_filter(nr.reduce_noise(audio_clip = np.asfortranarray(data_at), noise_clip = np.asfortranarray(noise_profile_at),
                                                                                              n_grad_freq = 6,
                                                                                              n_std_thresh = 1.5,
                                                                                              prop_decrease = 1),
                                                                             100, fs),
                                                       45, 2, fs)

    # Perform the filtering on the recordings of the AudiTechnica microphone
    filt_data_shure = butter_notch_filter(butter_highpass_filter(nr.reduce_noise(audio_clip = np.asfortranarray(data_shure), noise_clip = np.asfortranarray(noise_profile_shure),
                                                                                              n_grad_freq = 6,
                                                                                              n_std_thresh = 1.5,
                                                                                              prop_decrease = 1),
                                                                             100, fs),
                                                       45, 2, fs)

    # Extract the filename of the currently processed raw file, and use regex. to find it's associated number
    filename = os.path.basename(files[i])
    file_no = int(re.findall('[0-9]+', filename)[0])

    # I'll split for each category by it's associated number: 1-16 will be training data, 17-18 will be validation data and 19-20 will be test data
    category = ''
    if int(file_no < 17): 
        category = 'train'
    elif int(file_no < 19): 
        category = 'validate'
    else: 
        category = 'test'

    # Write processed sound files to their designated place
    wavfile.write(os.path.join('data/at/' + category, filename ), fs, filt_data_at)
    wavfile.write(os.path.join('data/shure/'  + category, filename ), fs, filt_data_shure)

    # Report progress
    print('Processed file: ' + filename + '; file ' + str(i+1) + ' out of ' + str(n) +'.')