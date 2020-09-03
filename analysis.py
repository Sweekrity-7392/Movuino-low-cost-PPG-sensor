import numpy as np
import matplotlib.pylab as plt
import pandas as pd
# from sklearn import preprocessing
from scipy.signal import butter, filtfilt
import scipy.fftpack

def plotTimeSeries(data, key:list, colorlist:list, label:str):
    """ function to plot raw data from each channnel against time points
    data: dataframe
    key: list
    """
    for index in key:
        fig = plt.figure(figsize=[10, 4])
        gs = fig.add_gridspec(3, 1)
        channel = data[index].columns[1:]
        for c in range(len(channel)):
            ax = fig.add_subplot(gs[c, 0])
            ax.plot(data[index, 'timestamp'], data[index, channel[c]], color=colorlist[c])
            ax.set_ylabel('%s \n %s' %(label,channel[c]))
            ax.set_xticks([])
        ax.set_xticks(data[index, 'timestamp'])
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=45, horizontalalignment='right')
        ax.set_xlabel('Time points (ms)')

def signalToNoiseRation(data, key, colorlist):
    """
    1) first convert time series data to frequency using FFT
    2) compute noise level and signal level
    3) compute signal to noise ratio """
    Amplitude = []
    Frequency = []
    Signal_Noise = []
    fft_Data = data
    fig = plt.figure(figsize=[10, 4])
    gs = fig.add_gridspec(1, len(key))
    for index in range(len(key)):
        channel = data[key[index]].columns[1:]
        signal_amp = []
        signal_freq = []
        sig_noise = []
        for c in range(len(channel)):
            ax = fig.add_subplot(gs[0, index])
            fft_Data.loc[:, (key[index], channel[c])] = scipy.fftpack.fft(data.loc[:, (key[index], channel[c])])
            s = np.array(fft_Data.loc[:, (key[index], channel[c])])
            sig_noise_amp = 2 / len(data.loc[:, (key[index], 'timestamp')]) * np.abs(fft_Data.loc[:,
                                                                                      (key[index], channel[c])])
            sig_noise_freq = np.abs(scipy.fftpack.fftfreq(len(data.loc[:, (key[index], 'timestamp')]), 2 / 60))
            # data.loc[:, (index, channel[c])] = np.fft.fft(data.loc[:, (index,channel[c])])
            # filtered_data.loc[:, (index, channel[c])] = np.abs(filtered_data.loc[:, (index, channel[c])])

            ax.plot(sig_noise_freq, sig_noise_amp, label='%s' %channel[c])

            #print amplitude
            signal_amplitude = pd.Series(sig_noise_amp).nlargest(2).round(0).astype(int).tolist()
            signal_amp.append(signal_amplitude)

            #print frequency
            # Calculate Frequency Magnitude
            magnitudes = abs(s[np.where(sig_noise_freq >= 0)])
            # Get index of top 2 frequencies
            peak_frequency = np.sort((np.argpartition(magnitudes, -2)[-2:]) / 2)
            signal_freq.append(peak_frequency)
            SNR = 20* np.log10(signal_amplitude[0]/signal_amplitude[1])
            # SNR = signal_amplitude[0] - signal_amplitude[1]
            sig_noise.append(SNR)
        Signal_Noise.append(sig_noise)
        ax.set_xlabel("Frequency")
        ax.set_ylabel('Amplitude')
        ax.legend()
        Amplitude.append(signal_amp)
        Frequency.append(signal_freq)
        # print(Signal_Noise)
    return Amplitude, Frequency, Signal_Noise, fft_Data

#filtering high freq data
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    data = filtfilt(b, a, data)
    return data

if __name__ == "__main__":
    # key = ['Movuino', 'PolarOH1', 'PolarH10', 'Maxim']
    key = ['Movuino', 'PolarOH1']
    colorlist = ['r', 'c', 'g']
    data = pd.read_pickle('data.pkl')
    baseline_corrected = pd.read_pickle('baseline_corrected_data.pkl')
    # signalToNoiseRation(data, key, colorlist)
    # plt.show()
    a, f, e, fft_data = signalToNoiseRation(baseline_corrected, key, colorlist)
    plt.show()

    # Filter requirements.
    fs = 100  # sample rate, Hz
    order = 1  # sin wave can be approx represented as quadratic
    filtered_data = fft_data
    for index in range(len(key)):
        channel = fft_data[key[index]].columns[1:]
        for c in range(len(channel)):
            cutoff = f[index][c][1]  # desired cutoff frequency of the filter, Hz, slightly higher than actual 2 Hz
            # print(cutoff)
            # Filter the data, and plot filtered signals.
            filtered_data.loc[:, (key[index],channel[c])] = butter_lowpass_filter(
                fft_data.loc[:, (key[index],channel[c])], cutoff, fs, order)
    plotTimeSeries(filtered_data, key, colorlist, "filtered \n data")
    plt.show()
    filtered_data.to_pickle('filtered__Data.pkl')

