#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process the Arcelor dataset

Author: B. Giroux, using large portions of "Create dataset" and "Clustering" notebooks by A. Morin

"""
# necessary imports
import bz2
import datetime
import glob
import os
import pickle
import socket
import multiprocessing as mp

import numpy as np
from scipy import stats
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.dates import num2date

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from obspy.clients.filesystem.sds import Client
from obspy.core import UTCDateTime

import DataStructures
from DayOfData import SeismicData, save_the_waves
from Signal import Signal

# %%  Setup

if 'taiga' in socket.gethostname():
    base_folder = '/Volumes/Data2/Arcelor'
else:
    base_folder = '/Volumes/Arcelor'
main_folder = base_folder + '/sds'
out_folder = base_folder+'/output'

ds = DataStructures.SeisCompP()
_ = ds.scan_directory(main_folder)

try:
    os.makedirs(out_folder)
except OSError:
    pass

root_path = main_folder
client = Client(root_path)

dt = 3600 * 12
t0 = UTCDateTime('2018-09-07')
t1 = UTCDateTime('2022-03-13')

save_fmt = '%Y.%m.%d..%H.%M'

fs = 1000.0

N, Wn = signal.buttord(wp=125.0, ws=150.0, gpass=3.0, gstop=40.0, fs=fs)
filter_kw=dict(N=N, Wn=Wn, btype='lowpass', fs=fs, output='sos')
sos = signal.butter(**filter_kw)

do_mk_waves = True
do_comp_features = False
do_comp_features_pool = False
do_analysis = False
do_clustering = False
do_plot_traces = False

n_processes = 6

# %% Plotting parameters

colors = ['#66c2a5','#fc8d62','#8da0cb']
colormaps = [mcolors.LinearSegmentedColormap.from_list("my_custom_map", ['white', color]) for color in colors]
markers = ['.', '^', 'x']

def plot_ellipse(ax, xy, cov, **kwargs):
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    nconf = 2
    angle = np.rad2deg(np.arctan2(*v[:, np.argmax(abs(lambda_))][::-1]))
    width = lambda_[np.argmax(abs(lambda_))] * nconf * 2
    height = lambda_[1 - np.argmax(abs(lambda_))] * nconf * 2

    ell = Ellipse(xy=xy, width=width, height=height, angle=angle, **kwargs)
    ax.add_artist(ell)


def annotate_axs(axs, x=0.01, y=0.99):
    letters = 'abcdefghijklmnopqrstuvwxyz'

    if x <= 0.5:
        ha = 'left'
    else:
        ha = 'right'

    if y <= 0.5:
        va = 'bottom'
    else:
        va = 'top'

    for i, ax in enumerate(axs.flatten()):
        ax.text(x, y, f'{letters[i]})', ha=ha, va=va, transform=ax.transAxes)


# %%  Read data & get events

def process_batch_waves(batch):

    i = 1
    _nbatch = len(batch)
    for n in batch:
        t = t0 + n * dt

        print(f'Worker {os.getpid()} - Analyzing {t} to {t + dt}, {i} out of {_nbatch:.0f}', flush=True)
        sd = SeismicData(t, t + dt - 1 / fs, client)
        if sd.empty is True:
            continue

        windows = sd.multi_stalta(couples=((1, 60), (2.5, 60)),
                                  in_threshold=10,
                                  out_threshold=2,
                                  join_threshold=3,
                                  filter_kw=filter_kw)
        if windows is None:
            continue

        waves = sd.partition_signal(windows)
        if len(waves) == 0:
            continue

        path = os.path.join(out_folder,
                            t.datetime.strftime(save_fmt) + '_' + (t + dt).datetime.strftime(save_fmt) + '.bz2')
        save_the_waves(waves, path)
        i += 1

def mk_waves():
    nbatch = int((t1 - t0) / dt + 0.0000001)
    chunk_size = 5
    batch_chunks = [range(nbatch)[i:i + chunk_size] for i in range(0, nbatch, chunk_size)]

    # process_batch_waves(batch_chunks[0])
    with mp.Pool(processes=n_processes) as pool:
        pool.map(process_batch_waves, batch_chunks)

    # i = 1
    # for n in range(nbatch):
    #     t = t0 + n * dt
    #
    #     print(f'\rAnalyzing {t} to {t+dt}, {i} out of {nbatch:.0f}', end='')
    #     sd = SeismicData(t, t+dt - 1/fs, client)
    #     if sd.nsamples is None:
    #         continue
    #
    #     windows = sd.multi_stalta(couples=((1, 60), (2.5, 60)),
    #                                 in_threshold=10,
    #                                 out_threshold=2,
    #                                 join_threshold=3,
    #                                 filter_kw=filter_kw)
    #     if windows is None:
    #         continue
    #
    #     waves = sd.partition_signal(windows)
    #     if len(waves) == 0:
    #         continue
    #
    #     path = os.path.join(out_folder, t.datetime.strftime(save_fmt) + '_' + (t+dt).datetime.strftime(save_fmt) + '.bz2')
    #     save_the_waves(waves, path)
    #     i += 1

    with open(os.path.join(out_folder, 'config.p'), 'wb') as f:
        pickle.dump((Signal.geophones, Signal.xyz), f)    # save in case class variables were changed from default


# %% Compute features

def process_batch_features(batch_files):
    with open(os.path.join('/Volumes/Arcelor/output_lp', 'config.p'), 'rb') as f:
        tmp = pickle.load(f)
        Signal.geophones = tmp[0]
        Signal.xyz = tmp[1]

    _df = pd.DataFrame()
    for file in batch_files:
        with bz2.open(file, 'rb') as f:
            waves = pickle.load(f)

        for j, w in enumerate(waves):

            # print(f'\rFile {i + 1} out of {len(files)}, waveform {j + 1} out of {len(waves)}', end='')
            print(f'File {os.path.basename(file)}, waveform {j + 1} out of {len(waves)}', flush=True)
            if type(w) is not Signal:
                # no event detected
                continue

            try:
                tmp = w.statistics()
                if len(tmp) > 0:
                    _df = pd.concat([_df, tmp], ignore_index=True, axis=1)
            except Exception as e:
                # df = pd.concat([df, pd.Series()], ignore_index=True, axis=1)
                print(f'\n    Exception with {os.path.basename(file)}:', e, ', skipping')
    return _df


def comp_features(_files):
    df = pd.DataFrame()
    for i, file in enumerate(_files):
        with bz2.open(file, 'rb') as f:
            waves = pickle.load(f)

        for j, w in enumerate(waves):

            print(f'\rFile {i + 1} out of {len(_files)}, waveform {j + 1} out of {len(waves)}', end='')
            # print(f'File {i + 1} out of {len(files)}, waveform {j + 1} out of {len(waves)}')
            if type(w) is not Signal:
                # no event detected
                continue

            try:
                tmp = w.statistics()
                if len(tmp) > 0:
                    df = pd.concat([df, tmp], ignore_index=True, axis=1)
            except Exception as e:
                # df = pd.concat([df, pd.Series()], ignore_index=True, axis=1)
                print('\n    Exception', e, 'skipping')

    df = df.T

    df['DATE'] = pd.DatetimeIndex(df['DATE'])
    df.set_index('DATE', inplace=True, drop=True)
    df.to_csv(os.path.join(out_folder, 'waveform_data.csv'))

def comp_features_pool(_files):

    chunk_size = 100
    file_chunks = [_files[i:i + chunk_size] for i in range(0, len(_files), chunk_size)]

    with mp.Pool(processes=n_processes) as pool:
        list_of_dfs = pool.map(process_batch_features, file_chunks)

    df = pd.concat(list_of_dfs, ignore_index=True, axis=1)
    df = df.T

    df['DATE'] = pd.DatetimeIndex(df['DATE'])
    df.set_index('DATE', inplace=True, drop=True)
    df.to_csv(os.path.join(out_folder, 'waveform_data.csv'))


# %%  Analysis

def analysis():
    # we can just load the data in case it was already computed
    df = pd.read_csv(os.path.join(out_folder, 'waveform_data.csv'))
    # df.index = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])) - pd.Timedelta(hours=4)
    # df = df.drop('DATE', axis=1)

    Xdf = df.copy()
    Xdf = Xdf.drop('DATE', axis=1)
    # We first take the logarithm of a few selected variables... Any transform can be done. The world is your oyster in machine learning.
    energy = Xdf.columns.str.contains('ENERGY')
    other = Xdf.columns.isin(['DUR', 'RISETIME', 'envMAX', 'envAREA', 'dftAVG', 'dftMAX', 'dftVAR'])
    islog = energy | other

    Xdf.loc[:, islog] = np.log10(Xdf.loc[:, islog])
    Xdf = (Xdf - Xdf.mean())/Xdf.std()
    Xdf['DATE'] = df['DATE']
    Xdf.set_index('DATE', inplace=True, drop=True)
    Xdf = Xdf.dropna(how='any', axis=1)
    for key in Xdf.keys():
        if '_BEARING' in key and key.startswith('DA'):
            Xdf = Xdf.drop(key, axis=1)
        elif '_PLUNGE' in key and key.startswith('DA'):
            Xdf = Xdf.drop(key, axis=1)
    Xdf.to_csv(os.path.join(out_folder, 'classified_signals_pca_features.csv'))


    pca = PCA()
    X = pca.fit_transform(Xdf)
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c='k', alpha=0.3, s=3)
    plt.gca().set_aspect(1)
    # plt.xlim(-10, 15)
    # plt.ylim(-10, 30)
    plt.subplot(222)
    plt.scatter(X[:, 0], X[:, 2], c='k', alpha=0.3, s=3)
    plt.gca().set_aspect(1)
    plt.subplot(223)
    plt.scatter(X[:, 0], X[:, 3], c='k', alpha=0.3, s=3)
    plt.gca().set_aspect(1)
    plt.subplot(224)
    plt.scatter(X[:, 1], X[:, 2], c='k', alpha=0.3, s=3)
    plt.gca().set_aspect(1)
    plt.show()

# %% Clustering

def clustering():
    df = pd.read_csv(os.path.join(out_folder, 'classified_signals_pca_features.csv'))
    Xdf = df.copy()
    Xdf = Xdf.drop('DATE', axis=1)

    # remove outliers
    ii = (np.abs(stats.zscore(Xdf)) < 3).all(axis=1)
    Xdf = Xdf[ii]
    dates = df['DATE'][ii]

    pca = PCA()
    X = pca.fit_transform(Xdf)

    ncomponents = 25
    nclusters = 3
    np.random.seed(42069)
    clf = GaussianMixture(n_components=nclusters)
    labels = clf.fit_predict(X[:, :ncomponents])

    fig, axs = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(9, 5))

    xlim = [-16, 20]
    ylim = [-6, 10]
    axs[0].hist2d(X[:, 0], X[:, 1], 100, range=[xlim, ylim], norm=LogNorm(), cmap='Greys')
    axs[1].hist2d(X[:, 0], X[:, 1], 100, range=[xlim, ylim], norm=LogNorm(), cmap='Greys', zorder=0)
    # axs[0].hist2d(X[:, 0], X[:, 1], 100, norm=LogNorm(), cmap='Greys')
    # plt.show()

    patches = []
    counts = np.zeros(clf.n_components)

    covsize = np.prod(clf.covariances_[:, :2, :2][:, [0, 1], [0, 1]], axis=-1)
    covorder = covsize.argsort().argsort()

    for c in range(clf.n_components):
        cc = labels == c
        counts[c] = cc.sum()

        x, y = X[cc, :2].T

        xy = clf.means_[[0, 2, 1][c], :2]
        cov = clf.covariances_[[0, 2, 1][c], :2, :2]

        plot_ellipse(ax=axs[1], xy=(xy), cov=cov, fc=colors[c], ec=colors[c], lw=3, alpha=0.8,
                     zorder=1 + nclusters - covorder[[0, 2, 1][c]])
        patches.append(Rectangle([0, 0], 1, 1, lw=0, fc=colors[c], label=c))

    # axs[0].set_aspect(1)
    # axs[1].set_aspect(1)
    axs[1].set_xlim(*xlim)
    axs[1].set_ylim(*ylim)

    # axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))
    # axs[0].xaxis.set_major_locator(plt.MaxNLocator(4))

    axs[0].set_ylabel('PCA$_2$')
    axs[0].set_xlabel('PCA$_1$')
    axs[1].set_xlabel('PCA$_1$')
    axs[1].legend(handles=patches, loc='lower right', frameon=False, title='Cluster', ncol=1)
    axs[1].tick_params('y', length=0)
    # fig.subplots_adjust(wspace=0.0)

    bbox = axs[1].get_position()
    ax3 = fig.add_axes([bbox.x1 + 0.025 * (bbox.x1 - bbox.x0), bbox.y0, 0.05 * (bbox.x1 - bbox.x0), bbox.y1 - bbox.y0])
    ax3.yaxis.tick_right()
    ax3.set_xticks([])
    for c in range(clf.n_components):
        ax3.bar(0, counts[c], width=1, bottom=0 if c == 0 else counts[:c].sum(), fc=colors[c], ec='k')
    ax3.set_ylim(0, counts.sum())
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_yticks(counts.cumsum() - counts / 2)
    ax3.set_yticklabels([f'{100 * i:.0f}%'.zfill(3) for i in counts / counts.sum()])
    ax3.invert_yaxis()

    annotate_axs(axs, x=0.02, y=0.98)

    # fig.patch.set_facecolor('none')
    plt.savefig(os.path.join(out_folder, 'clusters_pca_features.pdf'))
    plt.show()

    dates_clusters = []
    for c in range(clf.n_components):
        cc = labels == c

        x0 = np.mean(X[cc, :ncomponents], axis=0)

        d = np.sqrt(np.sum((X[:, :ncomponents] - np.tile(x0, (X.shape[0], 1)))**2, axis=1))
        d[np.logical_not(cc)] = d.max()
        ind_x0 = np.argsort(d)
        dates_clusters.append(tuple(dates.values[ind_x0[:3]]))

    with open(os.path.join(out_folder, 'dates_clusters.p'), 'wb') as f:
        pickle.dump(dates_clusters, f)

# %%  traces for each cluster

def plot_traces():
    with open(os.path.join(out_folder, 'config.p'), 'rb') as f:
        tmp = pickle.load(f)
        Signal.geophones = tmp[0]
        Signal.xyz = tmp[1]

    with open(os.path.join(out_folder, 'dates_clusters.p'), 'rb') as f:
        dates_clusters = pickle.load(f)

    files = glob.glob(os.path.join(out_folder, '*.bz2'))

    for nc, dates_cluster in enumerate(dates_clusters):
        traces = []
        for d in dates_cluster:
            date, hour = d.split()
            date = date.replace('-', '.')
            if int(hour[:2]) >= 12:
                date += '..12'
            else:
                date += '..00'
            for fname in files:
                fname_short = os.path.basename(fname)
                if fname_short.startswith((date)):
                    print(fname_short)
                    with bz2.open(fname, 'rb') as f:
                        waves = pickle.load(f)

                    for w in waves:
                        if w.day == datetime.datetime.fromisoformat(d):
                            traces.append(w)
                    break

            t = traces[0]

            for n, g in enumerate(Signal.geophones):
                data = t.waves[n, :, :]
                if np.all(np.isnan(data[Signal.xyz.index('E'), :])) and np.all(np.isnan(data[Signal.xyz.index('N'), :])):
                    # for sure we have a 1C geophone
                    if np.all(np.isnan(data[Signal.xyz.index('Z'), :])):
                        continue
                    else:
                        plt.figure(figsize=(12, 4))
                        plt.plot(num2date(t.t), data[Signal.xyz.index('Z'), :], color='C1')
                        plt.plot(num2date(t.t), signal.sosfiltfilt(sos, data[Signal.xyz.index('Z'), :]),
                                 color='C0', alpha=0.5)
                        plt.title(f'{date[:10].replace('.', '-')} Cluster {nc} - ' + g + 'Z')
                        plt.xlabel('Time')
                        plt.tight_layout()
                        plt.show()#block=False)
                else:
                    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
                    ax[0].plot(num2date(t.t), data[Signal.xyz.index('E'), :], color='C1')
                    ax[0].plot(num2date(t.t), signal.sosfiltfilt(sos, data[Signal.xyz.index('E'), :]),
                               color='C0', alpha=0.5)
                    ax[0].set_title(f'{date[:10].replace('.', '-')} Cluster {nc} - ' + g + ' E' )

                    ax[1].plot(num2date(t.t), data[Signal.xyz.index('N'), :], color='C1')
                    ax[1].plot(num2date(t.t), signal.sosfiltfilt(sos, data[Signal.xyz.index('N'), :]),
                               color='C0', alpha=0.5)
                    ax[1].set_title(f'{date[:10].replace('.', '-')} Cluster {nc} - ' + g + ' N' )

                    ax[2].plot(num2date(t.t), data[Signal.xyz.index('Z'), :], color='C1')
                    ax[2].plot(num2date(t.t), signal.sosfiltfilt(sos, data[Signal.xyz.index('Z'), :]),
                               color='C0', alpha=0.5)
                    ax[2].set_title(f'{date[:10].replace('.', '-')} Cluster {nc} - ' + g + ' Z' )
                    ax[2].set_xlabel('Time')
                    plt.tight_layout()
                    plt.show()#block=False)


# %% main
if __name__ == '__main__':

    if do_mk_waves:
        mk_waves()

    if do_comp_features or do_comp_features_pool:
        with open(os.path.join(out_folder, 'config.p'), 'rb') as f:
            tmp = pickle.load(f)
            Signal.geophones = tmp[0]
            Signal.xyz = tmp[1]

        files = glob.glob(os.path.join(out_folder, '*.bz2'))

        if do_comp_features:
            comp_features(files)
        elif do_comp_features_pool:
            comp_features_pool(files)

    if do_analysis:
        analysis()

    if do_clustering:
        clustering()

    if do_plot_traces:
        plot_traces()
