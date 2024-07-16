import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import xarray as xr

import datetime
import pickle
import copy

import obspy
from obspy.clients.filesystem.sds import Client
from obspy.core import UTCDateTime

from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import butter, filtfilt
from obspy.core import Stream

import DataStructures as ds
from Signal import Signal
from matplotlib.dates import num2date

import pickle

def join_common_signals(windows, threshold=1):

    # function that checks if detected signals are in a common window
    # returns the union of those windows
    TRUE_WINDOWS = windows.copy()
    windows = np.vstack([w for w in TRUE_WINDOWS if len(w) != 0])
    windows = windows[windows.argsort(axis=0)[:, 0]]

    l0 = len(windows)
    lmax = l0
    l1 = -1
    i = 0

    while l1 != l0 or i < lmax:
        w = windows[0]
        close = np.any(np.abs(windows - w) < threshold, axis=1)
        in1 = np.prod(w[0] - windows, axis=1) < 0
        in2 = np.prod(w[1] - windows, axis=1) < 0
        contains = (w[0] - windows[:, 0]) * (w[1] - windows[:, 1]) < 0
        ii = close | in1 | in2 | contains
        subw = windows[ii]
        a = min(subw[:, 0])
        b = max(subw[:, 1])
        windows = windows[~ii]
        windows = np.vstack((windows, np.array((a, b))))
        l1 = l0
        l0 = len(windows)
        i += 1

    return windows


def save_the_waves(waves, path):
    with open(path, 'wb') as file:
        pickle.dump(waves, file)


class SeismicData:

    def __init__(self, t0, t1, client):

        self.client = client
        self.stations = self._parse_stations()
        self.waves = []

        self.t0 = t0
        self.t1 = t1

        self.frequency = None
        self.times = None

        self._load_data()

    def multi_stalta(self, couples, in_threshold, out_threshold, join_threshold, filter_kw):

        from scipy.signal import butter, filtfilt
        b, a = butter(**filter_kw)

        # here we iterate over every station's traces to compute sta-lta, storing the stalta in a list
        signals = []
        for tag, traces in self.stations.items():

            ftraces = filtfilt(b, a, traces).astype(np.int64)

            # stalta computation
            cf = np.sum(np.abs(ftraces, dtype=np.int64), axis=0, dtype=np.int64)

            for (sta, lta) in couples:
                nsta = sta * self.frequency
                nlta = lta * self.frequency
                signal = classic_sta_lta(cf, nsta, nlta)
                signals.append(signal)

        separate_windows = []
        for stalta in signals:
            subw = trigger_onset(stalta, in_threshold, out_threshold)
            separate_windows.append(subw)

        windows = join_common_signals(separate_windows, join_threshold*self.frequency)
        index = np.argsort(windows[:, 0])
        windows = windows[index]

        return windows

    def partition_signal(self, windows, offset_length=1.0):

        # this will not work if not every station has the same amount of components!!!
        traces = np.array([trace[1] for trace in sorted(self.stations.items())])

        waves = []
        for i, j in windows:
            offset = int((j - i) * offset_length)
            subi, subj = max(0, i - offset), min(traces.shape[-1], j + offset)
            waveform = traces[:, :, subi:subj]
            wave = Signal(self.time[subi:subj], waveform, num2date(self.time[subi]), rockfall_times=(i - subi, j - subi))
            waves.append(wave)
        return waves

    def _parse_stations(self):
        stations = {}
        channels = self.client.get_all_nslc()

        for tuple in channels:
            network, channel, _, component = tuple
            if channel not in stations:
                stations[channel] = {}

            stations[channel][component] = tuple

        return stations

    def _load_data(self):

        for station, dicts in self.stations.items():
            data = []
            for component, tmp in dicts.items():
                trace = self.client.get_waveforms(*tmp, self.t0, self.t1)
                data.append(trace[0].data)

            data = np.array(data)
            data -= np.mean(data, axis=1).astype(data.dtype)[:, None]
            self.stations[station] = data
        self.frequency = trace[0].meta.sampling_rate
        self.time = trace[0].times('matplotlib')


if __name__ == '__main__':

    root_path = r'G:\\contenu_D\\Gros-Morne\\Pegasus'
    client = Client(root_path)

    t0 = UTCDateTime('2020-07-18')
    t1 = UTCDateTime('2021-10-27')

    dt = 3600 * 1
    i = 1
    ti = t0 + dt * i

    while ti < t1:
        sd = SeismicData(t0, ti - 1/1000, client)
        windows = sd.multi_stalta(couples=((1, 60), (2.5, 60)),
                                  in_threshold=10,
                                  out_threshold=2,
                                  join_threshold=3,
                                  filter_kw=dict(N=5, btype='highpass', Wn=50, fs=1000))
        waves = sd.partition_signal(windows)

        save_the_waves(waves, )

        t0 = ti
        ti += dt



