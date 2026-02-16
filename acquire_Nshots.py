#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created on Sunday Dec 15, 2024
# Author @ Chris Niemann

import epics
import time
import h5py
import numpy as np
import os
import datetime
from p4p.client.thread import Context

## Create P4P context for cameras
ctx = Context('pva')  # uses environment by default (useenv=True)

# Define PVs to be saved for each shot
scalars = ['Motor12:PositionRead',
           '13PICAM1:cam1:IntensifierGain_RBV',
           '13PICAM1:cam1:RepetitiveGateDelay_RBV',
           '13PICAM1:cam1:RepetitiveGateWidth_RBV',
           '13PICAM1:cam1:Temperature_RBV',
           '13PICAM1:cam1:TemperatureActual',
           '13PICAM1:cam1:CleanBeforeExposure_RBV',
           '13PICAM1:cam1:CleanCycleCount_RBV',
           '13PICAM1:cam1:CleanCycleHeight_RBV',
           '13PICAM1:cam1:CleanSectionFinalHeight_RBV',
           '13PICAM1:cam1:CleanSerialRegister_RBV',
           '13PICAM1:cam1:CleanUntilTrigger_RBV',
           '13PICAM1:cam1:CleanSectionFinalHeightCount_RBV',
           'LAPD-TS-digitizer:Ch1:MaxVoltage',
           'LAPD-TS-digitizer:Ch2:MaxVoltage',
           'LAPD-TS-digitizer:Ch2:Calibration',
           'LAPD-TS-digitizer:Ch2:Energy',
           'LAPD-TS-digitizer:Period',
           'TS:InputSlit',
           'TS:IntermediateSlit',
           #'BNC4:chA:DelayRead',
           'TS:redchi_pos',
           'TS:fit_width',
           'TS:redchi_spectrum',
           'TS:sig_int',
           'TS:best_pos',
           'TS:raw_e_density',
           'TS:corrected_e_density',
           'TS:Te',
           'TS:Tmax',
           'TS:Tmin',
           'TS:area',
           'TS:fwhm',
           'TS:width_err',
           ]

arrays = ['LAPD-TS-digitizer:Time',
          'LAPD-TS-digitizer:Ch1:Trace',
          'LAPD-TS-digitizer:Ch2:Trace',
          ]

images = ['13PICAM1:Pva1:Image',  # TS picam
          ]  

def trigger(pvname=None, value=None, char_value=None, **kws):
    global TrigState
    TrigState=1
    
def ReadEpicsImage2(pv):
    try:
        image = ctx.get(pv)  # returns NumPy array directly, no metadata
        TimeStamp = time.time()
        return image, TimeStamp 
    except Exception as e:
        print(f"Error reading PV '{pv}': {e}")
        return None, None


def get_unique_filename(directory, filename):
    """Returns a unique filename in the specified directory."""
    base, ext = os.path.splitext(filename)
    full_path = os.path.join(directory, filename)

    i = 1
    while os.path.exists(full_path):
        new_filename = f"{base}_{i}{ext}"
        full_path = os.path.join(directory, new_filename)
        i += 1

    return full_path

    
    
if __name__ == "__main__":
    N=100      # number of shot to be recorded
    filename='ts'
    directory='./'
    
    # Define trigger:
    epics.PV("13PICAM1:cam1:ArrayCounter_RBV", callback=trigger)
    #epics.PV("LAPD-TS-digitizer:Ch1:Trace", callback=trigger)
    #epics.PV("phoeniX:epoch", callback=trigger) # internal 1 Hz trigger

    # modify filename to add date and make sure not to overwrite existing
    current_date = datetime.date.today()
    date_string = current_date.strftime("-%Y-%m-%d")
    filename= "".join([filename, date_string,".h5"])
    filename = get_unique_filename(directory,filename)

    # start camera acquisition
    epics.caput('13PICAM1:cam1:Acquire',1)   
    time.sleep(0.025)

    # open hdf5
    with h5py.File(filename, 'w') as file:
        tsgroup = file.create_group('timestamps') #use optional group for readability
        # create empty datasets to store scalars repeatedly N times
        scalar_pvs = {}
        for scalar in scalars:
            # Create a single PV object to access the data 
            scalar_pvs[scalar] = epics.PV(scalar)
            # Create datasets to save scalars
            file.create_dataset(scalar, (N,), dtype=float)
            tsgroup.create_dataset(scalar +'.timestamp', (N,), dtype=float)  # for timestamps
        file.create_dataset('epoch', (N,), dtype=float)    # add one for time
        
        # create empty datasets to store arrays repeatedly N times
        array_pvs={}
        for array in arrays:
            # Create PV object to access the data 
            array_pvs[array] = epics.PV(array)
            # Create datasets
            array_sample = epics.caget(array) # read 1st array to determine length
            file.create_dataset(array, shape=(N, len(array_sample)), maxshape=(None, len(array_sample)), chunks=(1, len(array_sample)), dtype=float) #without the group 
            tsgroup.create_dataset(array+'.timestamp', (N,), dtype=float)  # for timestamps
        # create directory for image
        for image in images:
            file.create_group(image)  # create directory
            
  

    start_time = time.time()    # for total run duration
    shot=0  # shot counter
    try:
        TrigState=0 # reset trigger
        while shot < N:
            epics.ca.poll(evt=0.01)     # chek for new events
        
            # waiting for trigger
            if TrigState == 1:
                trigger_time=time.time()
                t0_acquisition=time.perf_counter()
                #os.system('clear')  # clear screen

                time.sleep(0.3) #to allow slow scope readout
                
                with h5py.File(filename, 'a') as file:
                    tsgroup = file['timestamps']
                    
                    # 1. read scalars and write to hdf
                    t0 = t0_acquisition
                    for scalar in scalars:
                        value = scalar_pvs[scalar].get()   # read pv value
                        tstamp = scalar_pvs[scalar].timestamp   # read timestamp
                        file[scalar][shot] = value   # write pv to hdf
                        tsgroup[scalar + '.timestamp'][shot] = tstamp     # write timestamp to hdf
                        t1 = time.perf_counter()
                        print(f"{shot:>5}/{N-1:<5} {tstamp-trigger_time:>13.1f} {scalar[:40]:<40} {value:<12.3g}, dT={(t1-t0)*1000:.3g} ms")        
                        t0=t1
                    file['epoch'][shot] = time.time()   # also save epoch time 
                    
                    # 3. read images and write to hdf
                    for image_name in images:
                        image, timestamp = ReadEpicsImage2(image_name)
                        dset = file[image_name].create_dataset(f"image {shot}", data=image)
                        dset.attrs['timestamp'] = timestamp
                        t1 = time.perf_counter()
                        print(f"{shot:>5}/{N-1:<5} {timestamp-trigger_time:>13.1f}  {image_name[:40]:<40} {str(image.shape):<12}, dT={(t1-t0)*1000:.3g} ms")
                        t0=t1
                    
                    # 3. read arrays and write to hdf; do it last, they take the longest to populate
                    for array in arrays:
                        vector = array_pvs[array].get()
                        tstamp = array_pvs[array].timestamp
                        file[array][shot, :]   = vector    # save data
                        tsgroup[array + '.timestamp'][shot] = tstamp    # save timestamp
                        t1 = time.perf_counter()
                        print(f"{shot:>5}/{N-1:<5} {tstamp-trigger_time:>13.1f}  {array[:40]:<40} {str(vector.shape):<12}, dT={(t1-t0)*1000:.3g} ms")
                        t0=t1  
                        
                
                shot+=1
                TrigState = 0   #reset
                print(f"\033[1;31mdT this acquisition: {(time.time()-trigger_time):.3g} s  \033[0m")
        
        file.close()
        print('_' * 77)
        print(f"\033[1;32mRun {filename} complete. Runtime {(time.time()-start_time)/60:.3g} minutes.\033[0m")
        print()
            
    except KeyboardInterrupt:
        print('program terminated')
        ctx.close() # close pva context
