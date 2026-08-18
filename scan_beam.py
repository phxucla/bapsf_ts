#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created on Sunday Dec 15, 2024
# Author @ Chris Niemann

import epics
import time
import h5py
import numpy as np
import os
import sys
import datetime
import socket
import json
import getpass

from p4p.client.thread import Context

ctx = Context('pva')  # uses environment by default (useenv=True)

# Define PVs to be saved for each shot
scalars = ['Motor12:PositionRead',
           '13PICAM1:cam1:ArrayCounter_RBV',
           '13PICAM1:cam1:TriggerSource_RBV',
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
           '13PICAM1:cam1:StopCleaningOnPreTrigger_RBV',
           '13PICAM1:Pva1:TimeStamp_RBV',
           '13PICAM1:cam1:PhosphorDecayDelay_RBV',
           '13PICAM1:cam1:PhosphorDecayDelayResolution_RBV',
           'LAPD-TS-digitizer:Ch1:MaxVoltage',
           'LAPD-TS-digitizer:Ch2:MaxVoltage',
           'LAPD-TS-digitizer:Ch2:Calibration',
           'LAPD-TS-digitizer:Ch2:Energy',
           'LAPD-TS-digitizer:Period',
           'TS:InputSlit',
           'TS:IntermediateSlit',
           'BNC4:Ch1:Delay_RBV',
           ]

arrays = ['LAPD-TS-digitizer:Time',
          'LAPD-TS-digitizer:Ch1:Trace',
          'LAPD-TS-digitizer:Ch2:Trace',
          ]

images = ['13PICAM1:Pva1:Image',  # TS
          ]

# EPICS PV cache
_pv_cache = {}

def get_pv(name, auto_monitor=False):
    if name not in _pv_cache:
        _pv_cache[name] = epics.PV(name, auto_monitor=auto_monitor)
    return _pv_cache[name]


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


def save_epics_metadata(hdf_obj, pv, timeout=0.2):
    try:
        pv.get_ctrlvars(timeout=timeout)

        desc_pv = get_pv(pv.pvname + ".DESC", auto_monitor=False)

        metadata = {
            'pvname'      : pv.pvname,
            'description': desc_pv.get(as_string=True, timeout=timeout) if desc_pv.connected or desc_pv.wait_for_connection(timeout=0.05) else None,
            'units'       : pv.units,
            'precision'   : pv.precision,
            'lower_ctrl_limit' : pv.lower_ctrl_limit,
            'upper_ctrl_limit' : pv.upper_ctrl_limit,
            'lower_alarm_limit': pv.lower_alarm_limit,
            'upper_alarm_limit': pv.upper_alarm_limit,
            'lower_warning_limit': pv.lower_warning_limit,
            'upper_warning_limit': pv.upper_warning_limit,
            'lower_disp_limit': pv.lower_disp_limit,
            'upper_disp_limit': pv.upper_disp_limit,
            'enum_strs'   : str(pv.enum_strs),
            'type'        : str(pv.type),
            'count'       : pv.count,
            'host'        : pv.host,
            'access'      : pv.access,
        }

        # optional long description
        long_pv = get_pv(pv.pvname + ":LongDescription.VAL$", auto_monitor=False)

        if long_pv.connected:
            long_description = long_pv.get(as_string=True, use_monitor=False, timeout=timeout)
            if long_description is not None:
                metadata["long_description"] = long_description

        # save only valid values
        for key, value in metadata.items():
            if value is not None:
                hdf_obj.attrs[key] = value

    except Exception as e:
        print(f"WARNING: failed metadata read for {pv.pvname}: {e}")


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
    positions=np.round(np.arange(0.34, 0.38, 0.01),3) # fiber position in cm
    repetitions=1 # per position, each repetition is 2 shots: ts & bg
    WAIT_FOR_SLOW_SCOPES = 0.3   # don't read PVs until slow scopes have acquired
    PV_SETTLE_TIMEOUT = 5.0     # max seconds to wait for inputPVs to reach commanded value
    os.makedirs(directory, exist_ok=True) # create it first

    # Define trigger
    trigger_pv = "13PICAM1:cam1:ArrayCounter_RBV" #TS
    #trigger_pv = "phoeniX:epoch" # internal 1 Hz trigger
    trigger_object = epics.PV(trigger_pv, callback=trigger)

    # build actionlist
    inputPVs    = ['Motor12:PositionInput']
    readbackPVs = ['Motor12:PositionRead']
    N = len(positions)*repetitions*2        # number of shot to be recorded
    matrix = np.zeros((N,1), dtype=float)

    print(f"positions: {len(positions)}, repetitions: {repetitions}, N: {N}")
    i=0
    for d in positions:
        for _ in range(repetitions):
            matrix[i,0]=d
            matrix[i+1,0]=d
            i+=2
    print(matrix)

    # Create all CA PVs up front so they can connect in parallel
    scalar_pvs   = {name: get_pv(name) for name in scalars}
    array_pvs    = {name: get_pv(name) for name in arrays}
    input_pvs    = {name: get_pv(name) for name in inputPVs}
    readback_pvs = {name: get_pv(name) for name in readbackPVs}
    acquire_pv   = get_pv('13PICAM1:cam1:Acquire')

    # Start metadata PV connections too
    for name in scalars + arrays:
        get_pv(name + ".DESC", auto_monitor=False)
        get_pv(name + ":LongDescription.VAL$", auto_monitor=False)

    # Give all CA PVs a short fixed time to connect in parallel
    t0 = time.time()
    while time.time() - t0 < 0.5:
        epics.ca.poll(evt=0.01)

    # The 0.5s window above is shared with dozens of scalar/array/metadata
    # PVs and is not guaranteed to be enough for any single one of them, but
    # inputPVs/readbackPVs/acquire are essential to the scan, so explicitly
    # block until they are connected before writing to them or trusting
    # their readback -- otherwise a put() issued before connection is
    # silently dropped.
    for name, pv in {**input_pvs, **readback_pvs}.items():
        if not pv.wait_for_connection(timeout=5.0):
            print(f"WARNING: {name} did not connect within 5s")
    if not acquire_pv.wait_for_connection(timeout=5.0):
        print(f"WARNING: {acquire_pv.pvname} did not connect within 5s")

    # start camera acquisition
    acquire_pv.put(1)
    time.sleep(0.025)

    # Initialize, i.e. set all controlPVs to the first desired value
    for p, name in enumerate(inputPVs):
        print(f"Set {name} to {matrix[0,p]}")
        input_pvs[name].put(matrix[0,p])

    # now wait until RBV is within 1% of requested value
    RBV = 0*matrix[0,:]    # create empty matrix that will be filled with RBVs

    # check control values are matched using np.allclose and relative tolerance
    t0_settle = time.time()
    while not np.allclose(matrix[0,:], RBV, rtol=1e-3):
        for p, name in enumerate(readbackPVs):
            RBV[p] = readback_pvs[name].get(timeout=0.9)
        print(f"{matrix[0,:]} vs {RBV}")

        if time.time() - t0_settle > PV_SETTLE_TIMEOUT:
            print(f"WARNING: inputPVs did not settle within {PV_SETTLE_TIMEOUT}s (wanted {matrix[0,:]}, got {RBV}); proceeding anyway")
            break

        time.sleep(0.25)

    print("Initialization complete")

    # modify filename to add date and make sure not to overwrite existing
    current_date = datetime.date.today()
    date_string = current_date.strftime("-%Y-%m-%d")
    filename= "".join([filename, date_string,".h5"])
    filename = get_unique_filename(directory,filename)

    # open hdf5
    with h5py.File(filename, 'w') as file:
        # file-level metadata
        file.attrs['created_unix_time'] = time.time()
        file.attrs['created_iso'] = datetime.datetime.now().isoformat()
        file.attrs['filename'] = filename
        file.attrs['N_requested'] = N
        file.attrs['trigger'] = trigger_pv
        file.attrs['WAIT_FOR_SLOW_SCOPES'] = WAIT_FOR_SLOW_SCOPES
        file.attrs['hostname'] = socket.gethostname()
        file.attrs['user'] = getpass.getuser()
        file.attrs['script_name'] = os.path.basename(sys.argv[0])
        file.attrs['script_path'] = os.path.abspath(sys.argv[0])

        file.attrs['scalar_pvs'] = json.dumps(scalars)
        file.attrs['array_pvs'] = json.dumps(arrays)
        file.attrs['image_pvs'] = json.dumps(images)
        file.attrs['input_pvs'] = json.dumps(inputPVs)
        file.attrs['readback_pvs'] = json.dumps(readbackPVs)

        tsgroup = file.create_group('timestamps') # use optional group for readability
        actiongroup = file.create_group('actionlist') # to save actionlist data

        # create empty datasets to store scalars repeatedly N times
        valid_scalars = []
        for scalar in scalars:
            pv = scalar_pvs[scalar]

            if not pv.connected:
                print(f"WARNING: skipping missing PV {scalar}")
                continue

            valid_scalars.append(scalar)
            print(pv)    # so we see if it crashes

            dset = file.create_dataset(scalar, (N,), dtype=float) # create datasets to save scalars
            save_epics_metadata(dset, pv)

            tsgroup.create_dataset(scalar +'.timestamp', (N,), dtype=float)  # for timestamps

        scalars = valid_scalars

        file.create_dataset('epoch', (N,), dtype=float)    # add one for time
        file.create_dataset('dT_this_acquisition', (N,), dtype=float) # one for dT this acquisition

        # create empty datasets to store arrays repeatedly N times
        valid_arrays = []
        for array in arrays:
            pv = array_pvs[array]

            if not pv.connected:
                print(f"WARNING: skipping missing array PV {array}")
                continue

            try:
                array_sample = pv.get(timeout=0.2) # read 1st array to determine length
                if array_sample is None or len(array_sample) == 0:
                    raise RuntimeError("None returned")
            except Exception:
                print(f"WARNING: skipping unreadable array PV {array}")
                continue

            valid_arrays.append(array)
            print(pv)   # so we see when it crashes

            # Create datasets
            dset = file.create_dataset(array, shape=(N, len(array_sample)), maxshape=(None, len(array_sample)), chunks=(1, len(array_sample)), dtype=float) #without the group
            save_epics_metadata(dset, pv)

            tsgroup.create_dataset(array+'.timestamp', (N,), dtype=float)  # for timestamps

        arrays = valid_arrays

        # create directory for image
        valid_images = []     # keep only image PVs that exist
        for image in images:
            try:
                test = ctx.get(image, timeout=0.5)
                if test is None:
                    print(f"WARNING: skipping missing image PV {image}")
                    continue
                valid_images.append(image)
                file.create_group(image)  # create directory
            except Exception as e:
                print(f"WARNING: skipping missing image PV {image}: {e}")
                continue

        images = valid_images

        # create empty datasets to store control values repeatedly N times
        for p, name in enumerate(inputPVs):
            dset = actiongroup.create_dataset(name, (N,), dtype=float)
            save_epics_metadata(dset, input_pvs[name])



    start_time = time.time()    # for total run duration
    # Shot = "This shot that just happened"
    # A better variable name would be "past_shot" or something to clearly
    # contrast with "next_shot" which is "past_shot+1"
    shot=0    # shot counter
    print("Waiting for trigger")
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
                time.sleep(WAIT_FOR_SLOW_SCOPES) # wait for all pvs to populate

                # FIRST, SAVE all scalars to the HDF file so we save the actual motor positions before they start to move for next shot
                with h5py.File(filename, 'a') as file:
                    tsgroup = file['timestamps']
                    actiongroup = file['actionlist']

                    # 1. read scalars and write to hdf
                    t0 = t0_acquisition
                    for scalar in scalars:
                        try:
                            value = scalar_pvs[scalar].get()                # read pv value
                            tstamp = scalar_pvs[scalar].timestamp            # read timestamp

                            if value is None:
                                print(f"WARNING: scalar PV returned None: {scalar}")
                                continue

                            file[scalar][shot] = value        # write pv to hdf
                            tsgroup[scalar + '.timestamp'][shot] = tstamp     # write timestamp to hdf

                            t1 = time.perf_counter()
                            print(f"{shot:>5}/{N-1:<5} {tstamp-trigger_time:>13.1f}  {scalar[:40]:<40} {value:<12.3g}, dT={(t1-t0)*1000:.3g} ms")
                            t0=t1

                        except Exception as e:
                            print(f"WARNING: failed scalar {scalar}: {e}")
                            continue

                    file['epoch'][shot] = time.time()    # also save epoch time

                    # 2. read images and write to hdf
                    for image_name in images:
                        image, timestamp = ReadEpicsImage2(image_name)
                        if image is None:
                            print(f"WARNING: image PV failed: {image_name}")
                            continue

                        dset = file[image_name].create_dataset(f"image {shot}", data=image)
                        dset.attrs['timestamp'] = timestamp
                        t1 = time.perf_counter()
                        print(f"{shot:>5}/{N-1:<5} {timestamp-trigger_time:>13.1f}  {image_name[:40]:<40} {str(image.shape):<12}, dT={(t1-t0)*1000:.3g} ms")
                        t0=t1

                    # 3. read arrays and write to hdf. Read them last, they take the longest to populate
                    for array in arrays:
                        try:
                            vector = array_pvs[array].get()
                            tstamp = array_pvs[array].timestamp

                            if vector is None:
                                print(f"WARNING: array PV returned None: {array}")
                                continue

                            file[array][shot, :]   = vector    # save data
                            tsgroup[array + '.timestamp'][shot] = tstamp    # save timestamp

                            t1 = time.perf_counter()
                            print(f"{shot:>5}/{N-1:<5} {tstamp-trigger_time:>13.1f}  {array[:40]:<40} {str(vector.shape):<12}, dT={(t1-t0)*1000:.3g} ms")
                            t0=t1

                        except Exception as e:
                            print(f"WARNING: failed array {array}: {e}")
                            continue

                    # 4. Write inputPV to dataset
                    for p, name in enumerate(inputPVs):
                        actiongroup[name][shot] = matrix[shot,p] # also write to hdf

                    file['dT_this_acquisition'][shot] = time.time()-trigger_time    # also save dT this acquisition

                set_pv_time = time.time()
                # Only set the PVs if this is not the last shot
                # Since there is no N+1 datapoint in the actionlist
                if next_shot < N:
                    for p, name in enumerate(inputPVs):
                        print(f"\033[34mSet {name} to {matrix[next_shot,p]} for shot {next_shot}\033[0m")
                        input_pvs[name].put(matrix[next_shot,p])
                time.sleep(0.2)

                # ==========================================================
                # Third, wait until all inputPVs have been set (e.g. motors)
                RBV = 0*matrix[shot,:]    # create empty matrix that will be filled with RBVs
                time_wait_for_pvs = time.time()
                if next_shot < N:
                    while not np.allclose(matrix[next_shot,:], RBV, rtol=1e-3):
                        for p, name in enumerate(readbackPVs):
                            RBV[p] = readback_pvs[name].get(timeout=0.9)
                        print(f"\033[34m{matrix[next_shot,:]} vs {RBV}\033[0m")

                        if time.time() - time_wait_for_pvs > PV_SETTLE_TIMEOUT:
                            print(f"\033[33mWARNING: inputPVs did not settle within {PV_SETTLE_TIMEOUT}s (wanted {matrix[next_shot,:]}, got {RBV}); proceeding anyway\033[0m")
                            break

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

    finally:
        ctx.close() # close pva context
