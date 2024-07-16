import numpy as np
import matplotlib.pyplot as plt
import obspy

import DataStructures
from DayOfData import DayOfData
from Signal import Signal

import datetime
import time

import matplotlib


main_folder = 'G:\contenu_D\Gros-Morne\\'

ds = DataStructures.SeisCompP()
_ = ds.scan_directory(r'G:\contenu_D\Gros-Morne\Pegasus')


def yearday_to_date(yearday):
    year = yearday[:4]
    day = yearday[4:]

    date = datetime.date(int(year), 1, 1) + datetime.timedelta(int(day)-1)
    return date


import os

out_folder = 'output'
try:
    os.makedirs(out_folder)
except OSError:
    pass

days = list(ds.catalogue.keys())[:1]

for i, day in enumerate(days):

    print(f'Analysing day {yearday_to_date(day)}, {i + 1}/{len(days)}')
    try:
        t0 = time.time()
        dod = DayOfData(day, ds)

        dod.partition_signal(1, 60, 10, 2)
        dt = time.time() - t0

        dod.waves[0].statistics()
        print(f'Found {len(dod.waves)} signals. Time taken: {dt:.2f} seconds.')

        dod.save_the_day(out_folder)
        print()
    except Exception as e:
        print(f'Error with {yearday_to_date(day)}: {e}')
        print()
