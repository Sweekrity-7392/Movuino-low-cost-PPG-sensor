import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn import preprocessing
from scipy.signal import butter, filtfilt

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
        # plt.setp(labels, rotation=45, horizontalalignment='right')
        ax.set_xlabel('Time points (ms)')

def signalToNoiseRation(data, key, colorlist):
    """
    1) first convert time series data to frequency using FFT
    2) compute noise level and signal level
    3) compute signal to noise ratio """
    for index in key:
        channel = channel = data[index].columns[1:]
        for c in range(len(channel)):
            data.loc[:, (index, channel[c])] = np.abs(np.fft.fft(data.loc[:, (index,channel[c])]))
            # filtered_data.loc[:, (index, channel[c])] = np.abs(filtered_data.loc[:, (index, channel[c])])
    plotTimeSeries(data, key, colorlist, 'FFT')

if __name__ == "__main__":
    # key = ['Movuino', 'PolarOH1', 'PolarH10', 'Maxim']
    key = ['Movuino']
    colorlist = ['r', 'c', 'g']
    data = pd.read_pickle('data.pkl')
    filtered_data = pd.read_pickle('filtered_Data.pkl')
    # signalToNoiseRation(data, key, colorlist)
    # plt.show()
    # signalToNoiseRation(filtered_data, key, colorlist)
    # plt.show()
    print(filtered_data.Movuino)