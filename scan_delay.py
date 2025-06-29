#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip install pvapy --break-system-packages
# https://bctwg.readthedocs.io/en/latest/source/demo/doc.demo.example_01.html

import epics
import time
import h5py
import numpy as np
import os
import datetime
from p4p.client.thread import Context

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
           '13PICAM1:Pva1:TimeStamp_RBV',
           '13PICAM1:cam1:PhosphorDecayDelay_RBV',
           '13PICAM1:cam1:PhosphorDecayDelayResolution_RBV',
           'LAPD-TS-digitizer:Ch1:MaxVoltage',
           'LAPD-TS-digitizer:Ch2:MaxVoltage',
           'LAPD-TS-digitizer:Ch2:Calibration',
           'LAPD-TS-digitizer:Ch2:Energy',
           'LAPD-TS-digitizer:Period_RBV',
           'TS:InputSlit',
           'TS:IntermediateSlit',
           'BNC3:chB:DelayRead',
           ]

arrays = ['LAPD-TS-digitizer:Time',
          'LAPD-TS-digitizer:Ch1:Trace',
          'LAPD-TS-digitizer:Ch2:Trace',
          ]

images = ['13PICAM1:Pva1:Image',  # TS
          ]  


def trigger(pvname=None, value=None, char_value=None, **kws):
    global TrigState
    TrigState=1

# pip3 install p4p
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
    filename ='test'
    directory='./'
    delays=np.round(np.arange(0., 0.03, 0.002),3) # bnc delay in s
    repetitions=1 # per delay, each repetition is 2 shots: ts & bg 

    # Define trigger
    #epics.PV("phoeniX:epoch", callback=trigger) # internal 1 Hz trigger
    #epics.PV("13PICAM2:cam1:ArrayCounter_RBV", callback=trigger) #LIF
    epics.PV("13PICAM1:cam1:ArrayCounter_RBV", callback=trigger) #TS
    #epics.PV("PNGdigitizer:Ch1:Trace", callback=trigger)

    # modify filename to add date and make sure not to overwrite existing
    current_date = datetime.date.today()
    date_string = current_date.strftime("-%Y-%m-%d")
    filename= "".join([filename, date_string,".h5"])
    filename = get_unique_filename(directory,filename)

   # build actionlist
    inputPVs    = ['BNC3:chB:DelayDesired']
    readbackPVs = ['BNC3:chB:DelayRead']
    N = len(delays)*repetitions*2        # number of shot to be recorded
    matrix = np.zeros((N,1), dtype=float)
    
    print(f"delays: {len(delays)}, repetitions: {repetitions}, N: {N}")
    print(delays)
    i=0
    for d in delays:
        for _ in range(repetitions):
            matrix[i,0]=d
            matrix[i+1,0]=d
            i+=2
    print(matrix)


    # start camera acquisition
    epics.caput('13PICAM1:cam1:Acquire',1)   
    time.sleep(0.025)

    # Initialize, i.e. set all controlPVs to the first desired value
    for p in range(len(inputPVs)):
        print(f"Set {inputPVs[p]} to {matrix[0,p]}")
        epics.caput(inputPVs[p], matrix[0,p])

    # now wait until RBV is within 1% of requested value
    RBV = 0*matrix[0,:]    # create empty matrix that will be filled with RBVs

    # check control values are matched using np.allclose and relative tolerance
    while not np.allclose(matrix[0,:], RBV, rtol=1e-3):
        for p in range(len(inputPVs)):
            RBV[p] = epics.caget(f"{readbackPVs[p]}", timeout=0.9)
        print(f"{matrix[0,:]} vs {RBV}")

        time.sleep(0.25)

    print("Initialization complete")

    # open hdf5
    with h5py.File(filename, 'w') as file:
        tsgroup = file.create_group('timestamps') # use optional group for readability
        actiongroup = file.create_group('actionlist') # to save actionlist data

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

        # create empty datasets to store control values repeatedly N times
        for name in inputPVs:
            actiongroup.create_dataset(name, (N,), dtype=float)



    start_time = time.time()    # for total run duration
    # Shot = "This shot that just happened"
    # A better variable name would be "past_shot" or something to clearly
    # contrast with "next_shot" which is "past_shot+1"
    shot=0    # shot counter
    try:
        TrigState=0    # reset trigger
        while shot < N:
            epics.ca.poll(evt=0.01)        # chek for new events
            time.sleep(0.01)    # add a slight delay to avoid busy-waiting

            next_shot = shot+1

            # waiting for trigger
            if TrigState == 1:
                trigger_time=time.time()
                t0_acquisition=time.perf_counter()
                time.sleep(0.3) # wait for all pvs to populate
                #os.system('clear') # clear screen

                # FIRST, SAVE all scalars to the HDF file so we save the actual motor positions before they start to move for next shot
                with h5py.File(filename, 'a') as file:
                    tsgroup = file['timestamps']
                    actiongroup = file['actionlist']

                    # 1. read scalars and write to hdf
                    t0 = t0_acquisition
                    for scalar in scalars:
                        value = scalar_pvs[scalar].get()                # read pv value
                        tstamp = scalar_pvs[scalar].timestamp            # read timestamp
                        #value=epics.caget(scalars[i])
                        file[scalar][shot] = value        # write pv to hdf
                        tsgroup[scalar + '.timestamp'][shot] = tstamp     # write timestamp to hdf
                        t1 = time.perf_counter()
                        print(f"{shot:>5}/{N-1:<5} {tstamp-trigger_time:>13.1f}  {scalar[:40]:<40} {value:<12.3g}, dT={(t1-t0)*1000:.3g} ms")
                        t0=t1
                    file['epoch'][shot] = time.time()    # also save epoch time

                    # 2. read images and write to hdf
                    for image_name in images:
                        image, timestamp = ReadEpicsImage2(image_name)
                        dset = file[image_name].create_dataset(f"image {shot}", data=image)
                        dset.attrs['timestamp'] = timestamp
                        t1 = time.perf_counter()
                        print(f"{shot:>5}/{N-1:<5} {timestamp-trigger_time:>13.1f}  {image_name[:40]:<40} {str(image.shape):<12}, dT={(t1-t0)*1000:.3g} ms")
                        t0=t1

                    # 3. read arrays and write to hdf. Read them last, they take the longest to populate
                    for array in arrays:
                        vector = array_pvs[array].get()
                        tstamp = array_pvs[array].timestamp
                        file[array][shot, :]   = vector    # save data
                        tsgroup[array + '.timestamp'][shot] = tstamp    # save timestamp
                        t1 = time.perf_counter()
                        print(f"{shot:>5}/{N-1:<5} {tstamp-trigger_time:>13.1f}  {array[:40]:<40} {str(vector.shape):<12}, dT={(t1-t0)*1000:.3g} ms")
                        t0=t1

                    # 4. Write inputPV to dataset
                    for p, name in enumerate(inputPVs):
                        actiongroup[name][shot] = matrix[shot,p] # also write to hdf

                set_pv_time = time.time()
                # Only set the PVs if this is not the last shot
                # Since there is no N+1 datapoint in the actionlist
                if next_shot < N:
                    for p, inputPV in enumerate(inputPVs):
                        print(f"\033[34mSet {inputPV} to {matrix[next_shot,p]} for shot {next_shot}\033[0m")
                        epics.caput(inputPV, matrix[next_shot,p])
                time.sleep(0.2)

                # ==========================================================
                # Third, wait until all inputPVs have been set (e.g. motors)
                RBV = 0*matrix[shot,:]    # create empty matrix that will be filled with RBVs
                time_wait_for_pvs = time.time()
                if next_shot < N:
                    while not np.allclose(matrix[next_shot,:], RBV, rtol=1e-3):
                        for p in range(len(inputPVs)):
                            RBV[p] = epics.caget(f"{readbackPVs[p]}", timeout=0.9)
                        print(f"\033[34m{matrix[next_shot,:]} vs {RBV}\033[0m")
                        time.sleep(0.1)

                print(f"All PVs set after: {(time.time() - set_pv_time)*1e3:.1f} ms, spent {(time.time() - time_wait_for_pvs)*1e3:.1f} ms waiting for PVs")

                shot+=1
                TrigState = 0    #reset
                print(f"\033[1;31mdT this acquisition: {(time.time()-trigger_time):.3g} s  \033[0m")

        print('_' * 77)
        print(f"\033[1;32mRun {filename} complete. Runtime {(time.time()-start_time)/60:.3g} minutes.\033[0m")
        print()


    except KeyboardInterrupt:
        print('program terminated')
        ctx.close() # close pva context
