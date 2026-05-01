#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script converts the Arcelor data from SEGY to mseed

The time series in this dataset are very fragmented.  Obspy deals with this using masked arrays; we follow this
approach.  Because masks cannot be saved with the data directly in seed files, we create channels to store them.

Author: B. Giroux
"""

import glob
import multiprocessing as mp
import re
import os

import numpy as np
import obspy
from obspy.io.mseed.util import get_start_and_end_time


segy_dir = '/Volumes/Data0/Arcelor/data'
sds_dir = '/Volumes/Data2/Arcelor/sds'

network = 'UQ'

n_processes = 4

def get_stn_infos(stn):

    with open(os.path.join(segy_dir, f'DA{stn}.sen'), 'r') as f:
        lines = f.readlines()
    name = []
    northing = []
    easting = []
    elevation = []
    geophone_type = []
    gain = []
    sensitivity = []
    component = []

    for ln in range(9, 15):
        tmp = lines[ln].split()
        name.append(tmp[0][:5].replace('-', ''))
        northing.append(float(tmp[1]))
        easting.append(float(tmp[2]))
        elevation.append(float(tmp[3]))
        geophone_type.append(tmp[5])
        gain.append(float(tmp[8]))
        sensitivity.append(float(tmp[9]))
        if float(tmp[13]) > 0:
            component.append('N')
        elif float(tmp[14]) > 0:
            component.append('E')
        else:
            component.append('Z')

    return name, northing, easting, elevation, geophone_type, gain, sensitivity, component

params_da = []
for n in range(1, 5):
    params_da.append(get_stn_infos(n))


def process_day(day_dirs):
    for day_dir in day_dirs:
        print(f'Worker {os.getpid()} - Working on {day_dir}')
        segy_files = []
        for hour in range(24):
            dirname = os.path.join(day_dir, f'{hour:02d}')
            if os.path.isdir(dirname):
                for f in glob.glob(os.path.join(dirname, '*.sgy')):
                    segy_files.append(f)

        segy_files = sorted(segy_files)
        m = re.match(r'\S+stn(\d)_5s/\S+Valcartier(\d{4})(\d{2})(\d{2})', segy_files[0])
        if m is None:
            m = re.match(r'\S+stn(\d)_5s/\S+Arcelor(\d{4})(\d{2})(\d{2})', segy_files[0])
        stn, year, month, day = m.groups()
        name, northing, easting, elevation, geophone_type, gain, sensitivity, component = params_da[int(stn)-1]

        ids = []
        for n in range(len(name)):
            ids.append(f'{network}.{name[n]}..GP{component[n]}')

        st = obspy.read(segy_files[0])
        if len(st.traces) == 12:   # some files have 12 traces instead of 6, with 0s for the first 6 traces
            if np.all(st.traces[0].data == 0):
                st.traces = st.traces[6:]
            else:
                st.traces = st.traces[:6]
        for nt in range(len(ids)):
            st.traces[nt].id = ids[nt]

        for file in segy_files:
            st2 = obspy.read(file)
            if len(st2.traces) == 12:
                if np.all(st2.traces[0].data == 0):
                    st2.traces = st2.traces[6:]
                else:
                    st2.traces = st2.traces[:6]
            for nt in range(len(ids)):
                    st2.traces[nt].id = ids[nt]
            st += st2

        st.merge(method=1)
        for tr in st.traces:
            while tr.stats.starttime.julday != tr.stats.endtime.julday:
                # for days with data until midnight, normally there is one sample too much (first of next day),
                # that sample should be removed to avoid pitfalls when processing data
                tr.trim(endtime=tr.stats.endtime - tr.stats.delta)

        seed_file = os.path.join(sds_dir, year)
        if not os.path.isdir(seed_file):
            os.makedirs(seed_file)
        seed_file = os.path.join(seed_file, network)
        if not os.path.isdir(seed_file):
            os.makedirs(seed_file)

        for tr in st.traces:
            seed_file = os.path.join(seed_file, tr.stats.station)
            if not os.path.isdir(seed_file):
                os.makedirs(seed_file)
            seed_file = os.path.join(seed_file, tr.stats.channel+'.D')
            if not os.path.isdir(seed_file):
                os.makedirs(seed_file)
            seed_file = os.path.join(seed_file, tr.id+'.D.'+year+f'.{tr.stats.starttime.julday:03d}')
            if not np.ma.is_masked(tr.data):
                try:
                    tr.write(seed_file, format='MSEED')
                except obspy.io.mseed.InternalMSEEDError as err:
                    print(f'Worker {os.getpid()} - InternalMSEEDError: {err}\n  Encoding {seed_file} as float32', flush=True)
                    tr.data = tr.data.astype(np.float32)
                    tr.write(seed_file, format='MSEED', encoding='FLOAT32')
            else:
                # masked arrays cannot be saved in mseed file, we have to save the mask separately
                mask = tr.data.mask
                tr.data = tr.data.filled()
                try:
                    tr.write(seed_file, format='MSEED')
                except obspy.io.mseed.InternalMSEEDError as err:
                    print(f'Worker {os.getpid()} - InternalMSEEDError: {err}\n  Encoding {seed_file} as float32', flush=True)
                    tr.data = tr.data.astype(np.float32)
                    tr.write(seed_file, format='MSEED', encoding='FLOAT32')

                print(f'  worker {os.getpid()} - Saving data in {seed_file}', flush=True)

                comp = tr.stats.channel[2]  # don't use ids, order might have changed after merge
                header = {'network': network, 'station': tr.stats.station, 'channel': f'MA{comp}',
                          'starttime': tr.stats.starttime,
                          'sampling_rate': tr.stats.sampling_rate}
                tr_m = obspy.Trace(mask.astype(np.int32), header=header)  # bool not accepted,
                seed_file_ma = os.path.join(sds_dir, year, network, tr_m.stats.station, 'MA'+comp+'.D')
                if not os.path.isdir(seed_file_ma):
                    os.makedirs(seed_file_ma)
                seed_file_ma = os.path.join(seed_file_ma, tr_m.id + '.D.' + year + f'.{tr_m.stats.starttime.julday:03d}')
                tr_m.write(seed_file_ma, format='MSEED')
                print(f'  worker {os.getpid()} - Saving mask in {seed_file_ma}', flush=True)

            # reset seed_file to value at start of loop
            seed_file = os.path.join(sds_dir, year, network)

def already_done(stn, year, month, day):
    # if we have a file for GPZ channel of 1st geophone, station was processed already
    starttime = obspy.UTCDateTime(year, month, day)
    fname = network+f'.DA{stn}1..GPZ.D.{year}.{starttime.julday:03d}'
    fname = os.path.join(sds_dir, str(year), network, f'DA{stn}1', 'GPZ.D', fname)
    if os.path.isfile(fname):
        try:
            # basic check to see if we have a valid file
            _ = get_start_and_end_time(fname)
            return True
        except:
            return False
    return False

if __name__ == '__main__':

    if not os.path.exists(sds_dir):
        os.makedirs(sds_dir)

    print('Building list of directories to process')
    # find 'day' directories that contain segy files
    day_dir = []
    for stn in range(1, 5):
        for year in range(2018, 2023):
            for month in range(1, 13):
                for day in range(1, 32):
                    dirname = os.path.join(segy_dir, f'stn{stn}_5s', str(year), f'{month:02d}', f'{day:02d}')
                    if os.path.isdir(dirname):
                        if not already_done(stn, year, month, day):
                            print(f'  stn{stn}_5s', str(year), f'{month:02d}', f'{day:02d}  added to list')
                            day_dir.append(dirname)
                        else:
                            print(f'  stn{stn}_5s', str(year), f'{month:02d}', f'{day:02d}  already done')

    chunk_size = 5
    dir_chunks = [day_dir[i:i + chunk_size] for i in range(0, len(day_dir), chunk_size)]

    print('Starting conversion')
    # process_day(day_dir)
    with mp.Pool(processes=n_processes) as pool:
        pool.map(process_day, dir_chunks)
