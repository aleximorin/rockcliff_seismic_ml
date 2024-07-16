#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for data access and management"""

import os
import shelve

class Channel:
    def __init__(self, description=None, units='-', notes=None):
        self.description = description
        self.units = units
        self.notes = notes


class Band:
    def __init__(self, type='-', sample_rate=None, corner_period=None):
        self.type = type
        self.sample_rate = sample_rate
        self.corner_period = corner_period


PEGASUS_CHANNELS = {
    # State-of-health channels
    'VEI': Channel('System input voltage', 'µV'),
    'VE1': Channel('System current', 'µA'),
    'VDT': Channel('Internal datalogger system temperature', 'µ°C'),
    'VE2': Channel('Sensor current (Sensor A and Sensor B)', 'µA'),
    'VM1': Channel('Sensor A SOH channel 1', 'µV', 'This value typically represents mass position.\n\
For Nanometrics seismometers:\n\
- VM1 = W axis\n\
Note that SOH channel 4 is only available on 4-channel Digital Recorder models.'),
    'VM2': Channel('Sensor A SOH channel 2', 'µV', 'This value typically represents mass position.\n\
For Nanometrics seismometers:\n\
- VM2 = V axis\n\
Note that SOH channel 4 is only available on 4-channel Digital Recorder models.'),
    'VM3': Channel('Sensor A SOH channel 3', 'µV', 'This value typically represents mass position.\n\
For Nanometrics seismometers:\n\
- VM3 = U axis'),
    'VM4': Channel('Sensor B SOH channel 1', 'µV', 'This value typically represents mass position.\n\
Note that SOH channel 4 is only available on 4-channel Digital Recorder models.'),
    # Timing/Clock/GNSS SOH
    'VCQ': Channel('Clock quality', 'Integer 1% / count', 'A heuristic time quality value.'),
    'ATU': Channel('Time uncertainty', 'µs',
                   'Predicted difference between the digitizer clock and the GNSS receiver.'),
    'VCE': Channel('Absolute clock phase error', 'µs',
                   'Measured difference between the digitizer clock and the GNSS receiver. When GNSS is off \
(see VST), this value will be 0.'),
    'VNS': Channel('GNSS number of satellites used',
                   notes='Number of satellites used in the timing solution. When GNSS is off (see VST), the \
last known value is preserved.'),
    'VLA': Channel('GNSS latitude', 'µ°', 'When GNSS is off (see VST), the last known value is preserved.'),
    'VLO': Channel('GNSS longitude', 'µ°', 'When GNSS is off (see VST), the last known value is preserved.'),
    'VEL': Channel('GNSS elevation', 'cm', 'When GNSS is off (see VST), the last known value is preserved.'),
    'VST': Channel('GNSS position fix status', notes='0 = no position fix, 1 = 2D fix only, 2 = 3D fix'),
    'VPL': Channel('PLL status', notes='0=no lock, 1=coarse lock, 2=fine lock, 3=free running'),
    'VAN': Channel('GNSS antenna status',
                   notes='0=OK, 1=no antenna (open circuit), 2=antenna short, 3=unknown status'),
    'VCO': Channel('VCXO control setpoint', 'DAC counts',
                   'Pegasus uses a 16-bit DAC for the VCXO control voltage setpoint. Therefore, this value \
ranges from 0 to 65535 inclusive. When GNSS is off (see VST), the last known value is \
preserved.')}


UQAR_CHANNELS = {
    'UTA': Channel('Température de l\'air', '°C'),
    'URH': Channel('Humidité relative de l\'air', '%'),
    'USI': Channel('Radiation solaire incidente (perpendiculaire à la paroi)', 'W/m2'),
    'USR': Channel('Radiation solaire réfléchie (perpendiculaire à la paroi)', 'W/m2'),
    'UWS': Channel('Vitesse du vent', 'm/s'),
    'UGS': Channel('Vitesse des rafales', 'm/s'),
    'UWD': Channel('Direction du vent', '°'),
    'UPR': Channel('Précipitations (pluie et neige non différenciées)', 'mm'),
    'UHS': Channel('Humidité relative en surface d\'un bloc de grès', '%'),
    'UHG': Channel('Humidité relative à 30 cm dans un bloc de grès', '%'),
    'UHL': Channel('Humidité relative à 30 cm dans un bloc de shale', '%'),
    'UWC': Channel('Teneur en eau à la surface de la roche', 'm3/m3'),
    'UWE': Channel('Humidité à la surface de la roche', '%'),
    'UE1': Channel('Température de surface enregistrée par l\'extensomètre 1', '°C'),
    'UE2': Channel('Température de surface enregistrée par l\'extensomètre 2', '°C'),
    'UE3': Channel('Température de surface enregistrée par l\'extensomètre 3', '°C'),
    'UE4': Channel('Température de surface enregistrée par l\'extensomètre 4', '°C'),
    'UE5': Channel('Température de surface enregistrée par l\'extensomètre 5', '°C'),
    'UTS': Channel('Température de surface (bloc de grès, THR)', '°C'),
    'UT0': Channel('Température de surface (géoprécision)', '°C'),
    'UTL': Channel('Température à 30 cm de profondeur (bloc de shale, THR)', '°C'),
    'UTG': Channel('Température à 30 cm de profondeur (bloc de grès, THR)', '°C'),
    'UT1': Channel('Température à 30 cm de profondeur (géoprécision)', '°C'),
    'UT2': Channel('Température à 60 em de profondeur (géoprécision)', '°C'),
    'UT3': Channel('Température à 90 cm de profondeur (géoprécision)', '°C'),
    'UT4': Channel('Température à 120 cm de profondeur (géoprécision)', '°C'),
    'UT5': Channel('Température à 150 cm de profondeur (géoprécision)', '°C'),
    'UT6': Channel('Température à 180 cm de profondeur (géoprécision)', '°C'),
    'UT7': Channel('Température à 210 cm de profondeur (géoprécision)', '°C'),
    'UT8': Channel('Température à 240 cm de profondeur (géoprécision)', '°C'),
    'UT9': Channel('Température à 270 cm de profondeur (géoprécision)', '°C'),
    'UTB': Channel('Température à 300 cm de profondeur (géoprécision)', '°C'),
    'UD1': Channel('Déplacements enregistrés par l\'extensomètre 1', 'mm'),
    'UD2': Channel('Déplacements enregistrés par l\'extensomètre 2', 'mm'),
    'UD3': Channel('Déplacements enregistrés par l\'extensomètre 3', 'mm'),
    'UD4': Channel('Déplacements enregistrés par l\'extensomètre 4', 'mm'),
    'UD5': Channel('Déplacements enregistrés par l\'extensomètre 5', 'mm')
}


class SeismicChannels:
    # from https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/
    band_codes = {
        'F': Band('-', '≥ 1000 to < 5000', '≥ 10 sec'),
        'G': Band('-', '≥ 1000 to < 5000', '< 10 sec'),
        'D': Band('-', '≥ 250 to < 1000', '< 10 sec'),
        'C': Band('_', '≥ 250 to < 1000', '≥ 10 sec'),
        'E': Band('Extremely Short Period', '≥ 80 to < 250', '< 10 sec'),
        'S': Band('Short Period', '≥ 10 to < 80', '< 10 sec'),
        'H': Band('High Broad Band', '≥ 80 to < 250', '≥ 10 sec'),
        'B': Band('Broad Band', '≥ 10 to < 80', '≥ 10 sec')
    }

    instrument_codes = {
        'H': 'High Gain Seismometer',
        'L': 'Low Gain Seismometer',
        'G': 'Gravimeter',
        'M': 'Mass Position Seismometer',
        'N': 'Accelerometer',
        'P': 'Geophone'
    }

    orientation_codes = {
        'Z': 'Traditional, vertical',
        'N': 'Traditional, North-South',
        'E': 'Traditional, East-West',
        'A': 'Triaxial',
        'B': 'Triaxial',
        'C': 'Triaxial',
        'T': 'Transverse',
        'R': 'Radial',
        '1': 'Orthogonal components but non traditional orientations',
        '2': 'Orthogonal components but non traditional orientations',
        '3': 'Orthogonal components but non traditional orientations',
        'U': 'Optional components',
        'V': 'Optional components',
        'W': 'Optional components'
    }

    @classmethod
    def is_seismic(cls, name):
        return name[0] in cls.band_codes and name[1] in cls.instrument_codes and name[2] in cls.orientation_codes


class DataStructure:
    """Base class for managing access to the data.
    
    Attributes
    ----------
    type : str
        type of data structure
        
    files : list of str
        list of data files
        
    catalogue : dict
        daily catalogue of data fields.  dict keys are the combination of 
        year+day (where both year & day are str).  Data fields are: 
        
            net : str
                Network code/identifier
            sta : str
                Station code/identifier
            loc : str
                Location identifier
            chan : str
                Channel code/identifier
            typ : str
                data type
            n : int
                index of corresponding data file in `files`     
    """

    def __init__(self, t):
        self.type = t
        self.files = None
        self.catalogue = None
        self.directory = None

    def get_fields(self, filename):
        """Get data fields from filename
        
        Parameters
        ----------
        filename : str
            name of file to extract fields from
            
        Returns
        -------
        tuple holding the following fields:
            str
                Network code/identifier
            str
                Station code/identifier
            str
                Location identifier
            str
                Channel code/identifier
            str
                data type
        """
        # this method should be overridden, given here just to document once
        pass

    def has_data(self, year, day, seismic_only=False):
        """Check if data exist for given day
        
        Parameters
        ----------
        year : int
            corresponding year
        day : int
            day of year (Jan 1st is 0)
        seismic_only : boolean
            ignore data other than seismic traces
            
        Returns
        -------
        int
            Number of stations with data
        """
        stations = set([])
        key = '{0:4d}{1:03d}'.format(year, day)
        if key in self.catalogue:
            for (_, sta, _, chan, _, _) in self.catalogue[key]:
                if seismic_only:
                    if SeismicChannels.is_seismic(chan):
                        stations.add(sta)
                else:
                    stations.add(sta)
        return len(stations)

    def get_number_of_stations(self, network=None, seismic_only=True):
        """Returns the number of stations in network

        Parameters
        ----------
        network : str
            name of network. if None is passed, returns all stations
        seismic_only : boolean
            ignore data other than seismic

        Returns
        -------
        int
            Nomber of stations
        """
        stations = set([])
        for key in self.catalogue:
            for (net, sta, _, chan, _, _) in self.catalogue[key]:
                if network is not None and net != network:
                    continue
                if seismic_only:
                    if SeismicChannels.is_seismic(chan):
                        stations.add(sta)
                else:
                    stations.add(sta)
        return len(stations)

    def is_data_file(self, filename):
        """Check data file conformity according to directory tree structure
        (file content is not examined)
        
        Parameters
        ----------
        filename : str
            name of file to check
        
        Returns
        -------
        bool
            True if file is data file
        """
        # this method should be overridden, given here just to document once
        pass

    @staticmethod
    def load(filename):
        """load previously saved instance.
        
        Parameters
        ----------
        filename : str
            name of file
            
        Returns
        -------
        DataStructure
            saved structure
        """
        d = shelve.open(filename, flag='r')
        t = d['type']
        files = d['files']
        catalogue = d['catalogue']
        directory = d['directory']
        d.close()

        if t == 'SDS':
            ds = SeisCompP()
            ds.files = files
            ds.catalogue = catalogue
            ds.directory = directory
        else:
            raise NotImplementedError('type not yet implemented')

        return ds

    def save(self, filename):
        """Save instance to file.
        
        Parameters
        ----------
        filename : str
            name of file
            
        Notes
        -----
        Instance is saved in a shelve file, which is not guaranteed to be portable
        """
        d = shelve.open(filename)
        d['type'] = self.type
        d['files'] = self.files
        d['catalogue'] = self.catalogue
        d['directory'] = self.directory
        d.close()

    def scan_directory(self, dir_name, level=0):
        """Scan directory tree for data files.
        
        Parameters
        ----------
        dir_name : string
            base of directory tree
        level : int
            level of directory to scan, if 0, directory is where scanning is starting
            
        Returns
        -------
        files : list of string
            list of data files
        """
        files = []
        self.catalogue = {}

        ftmp = os.listdir(dir_name)
        for f in ftmp:
            if os.path.isdir(dir_name+os.sep+f):
                tmp = self.scan_directory(dir_name+os.sep+f, level=level+1)
                for t in tmp:
                    files.append(t)
            elif os.path.isfile(dir_name+os.sep+f):
                if self.is_data_file(dir_name+os.sep+f):
                    files.append(dir_name+os.sep+f)

        if level == 0:
            self.directory = dir_name
            self.files = files
            for n, file in enumerate(files):
                net, sta, loc, chan, typ, year, day = self.get_fields(file)
                if year+day in self.catalogue:
                    self.catalogue[year+day].append((net, sta, loc, chan, typ, n))
                else:
                    self.catalogue[year+day] = [(net, sta, loc, chan, typ, n)]

        return files


class SeisCompP(DataStructure):
    """
    SeisComP Data Structure (SDS) 1.0

    Purpose
    
    Define a simple directory and file structure for data files. The SDS
    provides a basic level of standardization and portability when adapting
    data servers (AutoDRM, NetDC, etc.), analysis packages (Seismic Handler,
    SNAP, etc.) and other classes of software that need direct access to data
    files.
    
    The basic directory and file layout is defined as:
    
    <SDSdir>/Year/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.DAY
    Definitions of fields
    
    SDSdir  :  arbitrary base directory
    YEAR    :  4 digit year
    NET     :  Network code/identifier, up to 8 characters, no spaces
    STA     :  Station code/identifier, up to 8 characters, no spaces
    CHAN    :  Channel code/identifier, up to 8 characters, no spaces
    TYPE    :  1 characters indicating the data type, recommended types are:
        'D' - Waveform data
        'E' - Detection data
        'L' - Log data
        'T' - Timing data
        'C' - Calibration data
        'R' - Response data
        'O' - Opaque data
    LOC     :  Location identifier, up to 8 characters, no spaces
    DAY     :  3 digit day of year, padded with zeros
    
    The dots, '.', in the file names must always be present regardless if 
    neighboring fields are empty.
    
    Additional data type flags may be used for extended structure definition.
    
    from: https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html
    """

    def __init__(self):
        super().__init__('SDS')

    def is_data_file(self, filename):
        path_parts = filename.split(os.sep)
        file_parts = path_parts[-1].split('.')
        if len(file_parts) != 7:
            return False
        net, sta, _, chn, typ, year, _ = file_parts
        # check if we follow structure
        if path_parts[-2] == chn+'.'+typ and path_parts[-3] == sta and path_parts[-4] == net and path_parts[-5] == year:
            return True

    def get_first_day(self):
        """Return date of first file with data
        
        Returns
        -------
        (year, day_of_year) : tuple of two int
        
        Notes
        -----
        day_of_year starts at 1
        
        """
        if self.files is None:
            raise RuntimeError('File list not defined; scan a directory')
        elif len(self.files) == 0:
            raise RuntimeError('File list empty; no data in scanned directory')

        _, _, _, _, _, year, day = self.files[0].split(os.sep)[-1].split('.')
        first_day = (year, day)

        for file in self.files[1:]:
            _, _, _, _, _, year, day = file.split(os.sep)[-1].split('.')
            # day is padded with zeros, so string comparison holds
            if first_day[0] > year or (first_day[0] == year and first_day[1] > day):
                first_day = (year, day)
        return int(first_day[0]), int(first_day[1])

    def get_last_day(self):
        """Return date of last file with data
        
        Returns
        -------
        (year, day_of_year) : tuple of two int
        
        Notes
        -----
        day_of_year starts at 1
        
        """
        if self.files is None:
            raise RuntimeError('File list not defined; scan a directory')
        elif len(self.files) == 0:
            raise RuntimeError('File list empty; no data in scanned directory')

        _, _, _, _, _, year, day = self.files[0].split(os.sep)[-1].split('.')
        last_day = (year, day)

        for file in self.files[1:]:
            _, _, _, _, _, year, day = file.split(os.sep)[-1].split('.')
            if last_day[0] < year or (last_day[0] == year and last_day[1] < day):
                last_day = (year, day)
        return int(last_day[0]), int(last_day[1])

    def get_fields(self, filename):
        return filename.split(os.sep)[-1].split('.')


class BufferOfUniformData(DataStructure):
    """
    ${N}/${S}/${S}.${N}.${L}.${C}.${Y}.${J} for organizing dataset files where
    the parameters are defined as follows:
        ${N} is the network name
        ${S} is the station name
        ${L} is the location name
        ${C} is the channel name
        ${Y} is 4-digit year
        ${J} is Julian day (day of the year)
    """

    def __init__(self):
        super().__init__('BUD')

    def is_data_file(self, filename):
        path_parts = filename.split(os.sep)
        file_parts = path_parts[-1].split('.')
        if len(file_parts) != 6:
            return False
        sta, net, _, _, _, _ = file_parts
        if path_parts[-2] == sta and path_parts[-3] == net:
            return True

    def get_fields(self, filename):
        sta, net, loc, chan, _, _ = filename.split(os.sep)[-1].split('.')
        return net, sta, loc, chan, ''


if __name__ == '__main__':
    ds = SeisCompP()
    ds.scan_directory(r'F:\Forillon')
    ds.load_day(2023, 214)

    print(ds.get_first_day(), ds.get_last_day())
    print(ds.has_data(2020, 215))
    print(ds.has_data(2020, 15))
    print(ds.has_data(2020, 415))
