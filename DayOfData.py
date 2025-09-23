import bz2
import copy
import pickle

import numpy as np
from scipy.signal import butter, sosfiltfilt
from matplotlib.dates import num2date, date2num
import matplotlib.pyplot as plt

from obspy.clients.filesystem.sds import Client
from obspy.core import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset

from Signal import Signal


def join_common_signals(windows, threshold=1):

    # function that checks if detected signals are in a common window
    # returns the union of those windows
    TRUE_WINDOWS = windows.copy()
    windows = [w for w in TRUE_WINDOWS if len(w) != 0]
    if len(windows) == 0:
        return None
    windows = np.vstack(windows)
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
    with bz2.open(path, 'wb') as file:
        pickle.dump(waves, file)


class SeismicData:

    def __init__(self, t0, t1, client):

        self.client = client
        self.stations = self._parse_stations()
        self.waves = []

        self.t0 = t0
        self.t1 = t1

        self.xyz = None

        self.empty = True
        self._load_data()

    def multi_stalta(self, couples, in_threshold, out_threshold, join_threshold, filter_kw):

        sos = butter(**filter_kw)
        length_threshold = 0
        for (_, lta) in couples:
            if length_threshold < lta:
                length_threshold = lta

        separate_windows = []  # detection limits in UTC timestamps
        for stn, traces in self.stations.items():
            if len(traces) == 0:
                continue
            if len(traces) == 3:
                # 3C data
                work_traces = []
                if np.ma.is_masked(traces[0].data):
                    tmp0 = traces[0].split()
                    tmp1 = traces[1].split()
                    tmp2 = traces[2].split()
                    for nt in range(len(tmp0)):
                        if tmp0[nt].data.size < length_threshold * tmp0[nt].stats.sampling_rate:
                            continue
                        work_traces.append([tmp0[nt], tmp1[nt], tmp2[nt]])
                else:
                    work_traces.append(traces)

                for wtraces in work_traces:
                    ftraces = []
                    for tr in wtraces:
                        ftraces.append(sosfiltfilt(sos, tr.data))
                    cf = np.abs(ftraces[0]) + np.abs(ftraces[1]) + np.abs(ftraces[2])
                    for (sta, lta) in couples:
                        nsta = sta * wtraces[0].stats.sampling_rate
                        nlta = lta * wtraces[0].stats.sampling_rate
                        if len(cf) < nlta:
                            continue
                        stalta = classic_sta_lta(cf, nsta, nlta)
                        subw = trigger_onset(stalta, in_threshold, out_threshold)
                        subw_t = []
                        for n in range(len(subw)):
                            subw_t.append( [(wtraces[0].stats.starttime + subw[n][0] * wtraces[0].stats.delta).timestamp,
                                            (wtraces[0].stats.starttime + subw[n][1] * wtraces[0].stats.delta).timestamp] )
                            separate_windows.append(subw_t)

            else:
                # Z component
                work_traces = []
                if np.ma.is_masked(traces[0].data):
                    for tr in traces[0].split():
                        if tr.data.size < length_threshold * tr.stats.sampling_rate:
                            continue
                        work_traces.append(tr)
                else:
                    work_traces.append(traces[0])

                for tr in work_traces:
                    ftrace = sosfiltfilt(sos, tr.data)
                    for (sta, lta) in couples:
                        nsta = sta * tr.stats.sampling_rate
                        nlta = lta * tr.stats.sampling_rate
                        if len(ftrace) < nlta:
                            continue
                        stalta = classic_sta_lta(ftrace, nsta, nlta)
                        subw = trigger_onset(stalta, in_threshold, out_threshold)
                        subw_t = []
                        for n in range(len(subw)):
                            subw_t.append( [(tr.stats.starttime + subw[n][0] * tr.stats.delta).timestamp,
                                            (tr.stats.starttime + subw[n][1] * tr.stats.delta).timestamp] )
                            separate_windows.append(subw_t)

        #     try:
        #         ftraces = sosfiltfilt(sos, traces).astype(np.int64)
        #     except ValueError as e:
        #         print(e)
        #         continue
        #
        #     # stalta computation
        #     if ftraces.shape[0] == 1:
        #         cf = ftraces.flatten()
        #     else:
        #         # we have 3C data, take sum of absolute values of components
        #         cf = np.sum(np.abs(ftraces, dtype=np.int64), axis=0, dtype=np.int64)
        #
        #     for (sta, lta) in couples:
        #         nsta = sta * self.frequency
        #         nlta = lta * self.frequency
        #         signal = classic_sta_lta(cf, nsta, nlta)
        #         signals.append(signal)
        #
        # separate_windows = []
        # for stalta in signals:
        #     subw = trigger_onset(stalta, in_threshold, out_threshold)
        #     separate_windows.append(subw)

        if len(separate_windows) == 0:
            return None
        windows = join_common_signals(separate_windows, join_threshold)
        if windows is None:
            return None
        index = np.argsort(windows[:, 0])
        windows = windows[index]

        return windows

    def partition_signal(self, windows, offset_length=1.0):

        sampling_rate = 0
        t_min = UTCDateTime(2099, 1, 1)
        t_max = UTCDateTime(1999, 1, 1)
        for stn, traces in self.stations.items():
            for tr in traces:
                sampling_rate = tr.stats.sampling_rate
                if tr.stats.starttime < t_min:
                    t_min = tr.stats.starttime
                if tr.stats.endtime > t_max:
                    t_max = tr.stats.endtime

        dt = t_max - t_min
        nsamples = 1 + int(round(dt * sampling_rate))

        traces = np.full((len(self.stations), 3, nsamples), np.nan)
        for ns, stn in enumerate(self.stations):
            if len(self.stations[stn]) == 1:
                # Z component
                if self.stations[stn][0].data.size == nsamples:
                    traces[ns, 2, :] = self.stations[stn][0].data
                else:
                    dt = self.stations[stn][0].stats.starttime - t_min
                    i0 = int(round(dt * sampling_rate))
                    traces[ns, 2, i0:(i0+self.stations[stn][0].data.size)] = self.stations[stn][0].data
            elif len(self.stations[stn]) == 3:
                if self.stations[stn][0].data.size == nsamples:
                    for n in (0, 1, 2):
                        traces[ns, n, :] = self.stations[stn][n].data
                else:
                    dt = self.stations[stn][0].stats.starttime - t_min
                    i0 = int(round(dt * sampling_rate))
                    for n in (0, 1, 2):
                        traces[ns, n, i0:(i0+self.stations[stn][0].data.size)] = self.stations[stn][n].data

        Signal.geophones = self.stations.keys
        Signal.xyz = [comp[2] for comp in self.xyz]

        time = np.arange(nsamples) / sampling_rate
        time = (date2num(t_min.datetime) + time / 86400.0)

        waves = []
        for t_on, t_off in windows:
            i = int(round((t_on - t_min.timestamp) * sampling_rate))
            j = int(round((t_off - t_min.timestamp) * sampling_rate))

            offset = int((j - i) * offset_length)
            subi, subj = max(0, i - offset), min(nsamples, j + offset)
            waveform = traces[:, :, subi:subj]

            if num2date(time[subi]) is None:
                # time[subi] time is masked
                continue
            wave = Signal(time[subi:subj], waveform, num2date(time[subi]), rockfall_times=(i - subi, j - subi))
            waves.append(wave)
        return waves

    def _parse_stations(self):
        stations = {}
        all_nslc = self.client.get_all_nslc()

        self.network = all_nslc[0][0]

        for nslc in all_nslc:
            network, station, _, channel = nslc
            if network != self.network:
                raise Exception('Network mismatch')

            if station not in stations:
                stations[station] = []
        return stations

    def _load_data(self):

        for station in self.stations:
            st = self.client.get_waveforms(self.network, station, "*", "*", self.t0, self.t1)
            if len(st) == 0:
                continue
            self.empty = False
            if len(st) == 1:
                # Z component
                self.stations[station].append(st.traces[0])
            elif len(st) == 3:
                # E N Z traces without masks
                for tr in st.traces:
                    self.stations[station].append(tr)
                if self.xyz is None:
                    self.xyz = [tr.stats.channel for tr in st.traces]
            elif len(st) == 2:
                # we have a Z component + its mask
                i_z, i_mz = None, None
                for nt in range(len(st)):
                    if st.traces[nt].stats.channel == 'MAZ':
                        i_mz = nt
                    else:
                        i_z = nt
                tr = st.traces[i_z].copy()
                tr.data = np.ma.MaskedArray(tr.data, mask=st.traces[i_mz].data.astype(bool))
                self.stations[station].append(tr)
            else:
                if len(st) != 6:
                    raise Exception('Should have 6 traces')
                i_e, i_n, i_z, i_me, i_mn, i_mz = None, None, None, None, None, None
                for nt in range(len(st)):
                    if st.traces[nt].stats.channel == 'MAE':
                        i_me = nt
                    elif st.traces[nt].stats.channel == 'MAN':
                        i_mn = nt
                    elif st.traces[nt].stats.channel == 'MAZ':
                        i_mz = nt
                    elif st.traces[nt].stats.channel[2] == 'E':
                        i_e = nt
                    elif st.traces[nt].stats.channel[2] == 'N':
                        i_n = nt
                    elif st.traces[nt].stats.channel[2] == 'Z':
                        i_z = nt

                tr = st.traces[i_e].copy()
                tr.data = np.ma.MaskedArray(tr.data, mask=st.traces[i_me].data.astype(bool))
                self.stations[station].append(tr)

                tr = st.traces[i_n].copy()
                tr.data = np.ma.MaskedArray(tr.data, mask=st.traces[i_mn].data.astype(bool))
                self.stations[station].append(tr)

                tr = st.traces[i_z].copy()
                tr.data = np.ma.MaskedArray(tr.data, mask=st.traces[i_mz].data.astype(bool))
                self.stations[station].append(tr)
                if self.xyz is None:
                    self.xyz = [tr.stats.channel for tr in self.stations[station]]


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
