import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from scipy.signal import hilbert, butter, filtfilt, periodogram, spectrogram, find_peaks
from scipy.integrate import trapz
from scipy.stats import skew, kurtosis

import mplstereonet as pst


def yearday_to_date(yearday):
    year = yearday[:4]
    day = yearday[4:]

    date = datetime.date(int(year), 1, 1) + datetime.timedelta(int(day)-1)
    return date


def signaltonoise(a, axis=-1):
    a = np.asanyarray(a)
    m = np.max(np.abs(a), axis=axis)
    sd = a.std(axis=axis)
    return (abs(np.where(sd == 0, 0, m/sd)))


def polarize(ogtrace, noise_threshold=None, plot3d=False):
    # NEZ sorted trace to ENZ sorted trace
    trace = ogtrace[[1, 0, -1]]

    if noise_threshold is not None:
        mask = np.linalg.norm(trace, axis=0) < noise_threshold * np.std(trace)
        #trace = trace[:, mask]
    else:
        mask = np.ones(len(trace.T)).astype(bool)

    trace -= trace.mean(axis=-1).astype(trace.dtype)[:, None]
    cov = np.cov(trace[:, mask])
    vecs, vals, v = np.linalg.svd(cov)

    azimuth = (np.arctan2(vecs[0, 0], vecs[1, 0]) * 180 / np.pi) % 360
    eve = np.sqrt(vecs[0, 0] ** 2 + vecs[1, 0] ** 2)
    incidence = np.arctan2(eve, vecs[2, 0]) * 180 / np.pi % 180
    rectilinearity = 1 - np.sqrt(vals[1] / vals[0])
    planarity = 1 - 2 * vals[2] / (vals[1] + vals[0])

    l1, l2, l3 = v @ trace

    if plot3d:
        e, n, z = trace

        vals = 2 * np.sqrt(vals)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(f'AZIMUTH={azimuth:.2f}, INCIDENCE={incidence:.2f}\nRECTILINEARITY={rectilinearity:.2f}, PLANARITY={planarity:.2f}')
        ax.set_xlabel('E')
        ax.set_ylabel('N')
        ax.set_zlabel('Z')
        ax.scatter(e, n, z, alpha=0.25, c='k')
        arrows = ax.quiver(*np.zeros_like(vecs), *(vecs) * vals, color='r')

        theta = np.linspace(0, 2 * np.pi, 100)
        orders = [[0, 1, 2], [0, 2, 1], [2, 0, 1]][::-1]
        for i, v1 in enumerate(vals, start=0):
            for j, v2 in enumerate(vals[i + 1:], start=0):
                x1 = np.cos(theta) * v1
                x2 = np.sin(theta) * v2
                x3 = np.zeros_like(theta)

                k = 3 - i - j
                order = orders.pop()
                rotated = vecs @ np.array((x1, x2, x3))[order]
                ax.plot(*rotated, c='r')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        maxima = np.max((np.abs(xlim), np.abs(ylim), np.abs(zlim)))
        ax.set_xlim(-maxima, maxima)
        ax.set_ylim(-maxima, maxima)
        ax.set_zlim(-maxima, maxima)
        ax.set_box_aspect((maxima, maxima, maxima))

        trace = v @ trace
        fig, axs = plt.subplots(3, 1, sharex='col', sharey='col')
        t = np.arange(len(trace.T)) / 1000
        for i, ax in enumerate(axs):
            ax.plot(t, trace[i], c='k')
            axs[i].set_ylabel(f'$\\lambda_{i + 1}$', rotation=0)
            axs[i].grid()

        axs[0].set_title('Onde polarisée')
        axs[-1].set_xlabel('Temps (s)')

    return rectilinearity, planarity, azimuth, incidence, l1, l2, l3


class Signal:
    def __init__(self, t: np.array, waves: np.array, day: datetime.date, rockfall_times: tuple):

        self.t = t
        self.waves = waves
        self.day = day
        self.rockfall_times = rockfall_times

    def plot(self):

        # plots the signal and the frequency content

        fig, axs = plt.subplots(9, 2, sharex='col', figsize=(8, 9))
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        axs[0, 0].set_title('Signal', loc='left')
        axs[0, 1].set_title('Spectre', loc='right')
        axs[-1, 0].set_xlabel('Time (s)')
        axs[-1, 1].set_xlabel('Frequency (Hz)')
        axs = axs.reshape(3, 3, 2)
        x = (self.t - self.t[0]) * 3600 * 24
        freq, powers = periodogram(self.waves[:, :, self.rockfall_times[0]:self.rockfall_times[1]], 1000,
                                   scaling='spectrum')
        geophones = ('00368', '00380', '00400')
        xyz = ('N', 'E', 'Z')
        for i, station in enumerate(self.waves):
            subaxs = axs[i]

            for j, wave in enumerate(station):
                wax, pax = subaxs[j]
                wax.plot(x, wave, c='k', alpha=1)
                wax.set_ylabel(xyz[j], rotation=0, va='center', ha='center')
                maxima = np.max(np.abs(wax.get_ylim()))
                ylim = (-maxima, maxima)
                wax.fill_betweenx(ylim, x[self.rockfall_times[0]], x[self.rockfall_times[1]], fc='red', alpha=0.5)
                wax.set_ylim(ylim)
                pax.plot(freq, np.sqrt(powers[i, j]), c='k')
                pax.yaxis.tick_right()
                wax.grid()
                pax.grid()
                box = axs[i, j, 0].get_position()
                box.y0 = box.y0 - i * 0.05 + 0.075
                box.y1 = box.y1 - i * 0.05 + 0.075
                axs[i, j, 0].set_position(box)
                box = axs[i, j, 1].get_position()
                box.y0 = box.y0 - i * 0.05 + 0.075
                box.y1 = box.y1 - i * 0.05 + 0.075
                axs[i, j, 1].set_position(box)
            xhalf = (subaxs[0, 0].get_position().x0 + subaxs[0, -1].get_position().x1) / 2
            yhalf = subaxs[0, 1].get_position().y1 + 0.02
            plt.figtext(xhalf, yhalf, f'Station {geophones[i]}', ha='center', va='center', size='x-large')
        fig.align_ylabels(axs[:, :, 0])
        fig.align_ylabels(axs[:, :, 1])
        plt.text(0.01, 0.012, self.day.strftime('%Y-%m-%d %H:%M:%S'), transform=fig.transFigure)

        return axs

    def corrplot(self):
        from matplotlib import patheffects as pe
        polarized_waves = []
        for i, geo in enumerate(self.waves):
            if np.any(geo.std(axis=-1) == 0):
                continue
            rectilinearity, planarity, azimuth, incidence, l1, l2, l3 = polarize(geo)
            polarized_waves.append(np.array([l1, l2, l3]))
        polarized_waves = np.vstack(w for w in polarized_waves)
        cov = np.corrcoef(polarized_waves)
        fig, ax = plt.subplots(tight_layout=True)
        im = ax.imshow(cov, vmin=-1, vmax=1, alpha=1)
        major = np.arange(0, 9)
        minor = np.arange(0, 9, 3) - 0.5
        directions = ['N', 'E', 'Z', 'N', 'E', 'Z', 'N', 'E', 'Z']
        geophones = ['00368', '00380', '00400']
        ax.set_xticks(major)
        ax.set_yticks(major)
        ax.set_xticklabels(directions)
        ax.set_yticklabels(directions)
        ax.set_xticks(minor, minor=True)
        ax.set_yticks(minor, minor=True)
        ax.grid(which='minor', lw=3, c='w')
        ax.xaxis.tick_top()
        for g, x in zip(geophones, minor):
            ax.text(-1, x + 1.5, g, ha='center', va='center', rotation=90)
            ax.text(x + 1.5, -1.1, g, ha='center', va='center')
        ax.tick_params('both', which='both', length=0)
        for spine in ['top', 'left', 'bottom', 'right']:
            ax.spines[spine].set_visible(False)
        for i in range(9):
            for j in range(9):
                ax.text(i, j, f'{cov[i, j]:.2f}', ha='center', va='center',
                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], c='w')
        return fig, ax

    def statistics(self):

        # computes statistics on the traces of the signal
        # returns a dictionary

        fullwaves = np.vstack([w for w in self.waves])[:, self.rockfall_times[0]:self.rockfall_times[1]]
        data = {}

        groupedwaves = fullwaves.reshape(3, 3, -1)

        # polarity attributes
        recs, plans, azis, incs = [], [], [], []
        polarized_waves = []
        geophones = ['00368', '00380', '00400']
        for i, geo in enumerate(groupedwaves):
            if np.any(geo.std(axis=-1) == 0):
                continue
            rectilinearity, planarity, azimuth, incidence, l1, l2, l3 = polarize(geo)
            polarized_waves.append(np.array([l1, l2, l3]))

            data[f'{geophones[i]}_BEARING'] = azimuth
            data[f'{geophones[i]}_PLUNGE'] = incidence

            recs.append(rectilinearity)
            plans.append(planarity)
            azis.append(azimuth)
            incs.append(incidence)

        polarized_waves = np.vstack([w for w in polarized_waves])

        MEAN_RECTILINEARITY = np.mean(recs)
        VAR_RECTILINEARITY = np.var(recs)

        MEAN_PLANARITY = np.mean(plans)
        VAR_PLANARITY = np.var(plans)

        strikes, dips = pst.plunge_bearing2pole(incs, azis)
        PLANE_STRIKE, PLANE_DIP = pst.fit_girdle(strikes, dips)

        (MEAN_PLUNGE, MEAN_BEARING), R_VALUE = pst.find_mean_vector(incs, azis)

        # geophones correlation attributes
        corrmat = np.corrcoef(polarized_waves)
        corrmat = corrmat[np.tril_indices_from(corrmat, -1)]
        corrmat = corrmat[~np.isnan(corrmat)]

        corrMAX = corrmat.max()
        corrAVG = corrmat.mean()
        corrMED = np.median(corrmat)
        corrSTD = np.std(corrmat)
        corrMIN = corrmat.min()
        corrRANGE = corrMAX - corrMIN

        abscorrmat = np.abs(corrmat)
        abscorrMAX = abscorrmat.max()
        abscorrAVG = abscorrmat.mean()
        abscorrMED = np.median(abscorrmat)
        abscorrSTD = np.std(abscorrmat)
        abscorrMIN = np.min(abscorrmat)
        abscorrRANGE = abscorrMAX - abscorrMIN

        crosscorrelations = []
        for i, sw1 in enumerate(groupedwaves):
            for j, sw2 in enumerate(groupedwaves[i + 1:], start=1):
                for dim in range(3):
                    if sw1[dim].sum() != 0 and sw2[dim].sum() != 0:
                        cc = np.correlate(sw1[dim], sw2[dim], mode='full')[int(len(fullwaves.T)) - 1:]
                        crosscorrelations.append(np.argmax(cc))

        """corrlagMAX = np.max(crosscorrelations)
        corrlagMIN = np.min(crosscorrelations)
        corrlagAVG = np.mean(crosscorrelations)
        corrlagMED = np.median(crosscorrelations)
        corrlagSTD = np.std(crosscorrelations)"""

        # we take the wave with the strongest signal to noise ratio
        waves = fullwaves[signaltonoise(fullwaves).argmax()]
        waves = waves[None, :]  # lazy not rewriting the code

        # basic statistics relative to some boring paper
        DUR = len(waves.T)

        envl = np.abs(hilbert(waves, axis=-1))

        envNPEAKS = len(find_peaks(envl[0], height=0.75*envl.max())[0])
        RISETIME = np.argmax(envl, axis=-1).mean()
        envMAX = np.max(envl, axis=-1).mean()
        envAREA = trapz(envl, dx=1 / 1000, axis=-1).mean()
        envAVG = np.mean(envl, axis=-1).mean()

        STN = signaltonoise(waves).mean()
        envKURT = kurtosis(waves, axis=1).mean()
        envSKEW = skew(waves, axis=1).mean()

        # filtered bandwidths attributes
        frequencies = [1] + list(np.arange(50, 500, 50)) + [499]
        for i in range(len(frequencies)-1):
            freqs = (frequencies[i], frequencies[i+1])
            w = np.array(freqs) * 2 / 1000
            b, a = butter(5, w, 'bandpass')
            filtered = filtfilt(b, a, waves)
            energy = (filtered**2).sum(axis=-1).mean()
            data[f'ENERGY_BP_{freqs[0]:.0f}-{freqs[1]:.0f}'] = energy
            data[f'KURT_BP_{freqs[0]:.0f}-{freqs[1]:.0f}'] = kurtosis(filtered, axis=-1).mean()

            b, a = butter(5, w, 'bandstop')
            filtered = filtfilt(b, a, waves)
            energy = (filtered ** 2).sum(axis=-1).mean()
            data[f'ENERGY_BS_{freqs[0]:.0f}-{freqs[1]:.0f}'] = energy
            data[f'KURT_BS_{freqs[0]:.0f}-{freqs[1]:.0f}'] = kurtosis(filtered, axis=-1).mean()

        # spectral attributes
        frequencies, amplitude = periodogram(waves, 1000, scaling='spectrum')
        dftAVG = amplitude.mean()
        dftMAX = amplitude.max(axis=1).mean()
        dftARGMAX = frequencies[amplitude.argmax(axis=1)].mean()
        dftVAR = amplitude.var(axis=1).mean()
        dftNPEAKS = len(find_peaks(amplitude[0], height=0.75 * dftMAX)[0])

        # DFT energy
        n = len(frequencies)
        quantiles = np.arange(0, 1.01, 0.1)
        for q in range(len(quantiles) - 1):
            i = int(quantiles[q] * n)
            j = int(quantiles[q + 1] * n)
            suby = amplitude[:, i:j]
            data[f'dftENERGY_{quantiles[q]}-{quantiles[q+1]}'] = suby.sum(axis=1).mean()

        DATE = self.day

        # keeps every variables named with uppercase letters to a dictionary
        variable_dict = locals().copy()
        for key, variable in variable_dict.items():
            for l in key:
                if l.isupper():
                    data[key] = variable  # here we add the key and the variable to the dict if the key has capitals
                    break

        return pd.Series(data)

