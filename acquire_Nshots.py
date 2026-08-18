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

## Create P4P context for cameras
ctx = Context('pva', conf={
    'iface_list': '10.97.106.4',
    'auto_addr_list': '0',
    'addr_list': '10.97.106.3,10.97.106.4,10.97.106.5'
})

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
           'LAPD-TS-digitizer:Ch1:MaxVoltage',
           'LAPD-TS-digitizer:Ch2:MaxVoltage',
           'LAPD-TS-digitizer:Ch2:Calibration',
           'LAPD-TS-digitizer:Ch2:Energy',
           'LAPD-TS-digitizer:Period',
           'TS:InputSlit',
           'TS:IntermediateSlit',
           'BNC4:Ch1:Delay_RBV',
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

# EPICS PV cache
_pv_cache = {}

def get_pv(name, auto_monitor=False):
    if name not in _pv_cache:
        _pv_cache[name] = epics.PV(name, auto_monitor=auto_monitor)
    return _pv_cache[name]


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
    N=3      # number of shot to be recorded
    filename='test'
    directory='./'
    WAIT_FOR_SLOW_SCOPES = 0.3   # don't read pVs until slow scopes have acquired but don't exceed 1s total period
    os.makedirs(directory, exist_ok=True) # create it first

    #trigger_pv = "phoeniX:epoch"  # internal 1 Hz trigger
    trigger_pv = "LAPD-TS-digitizer:Ch1:Trace"    
    trigger_object = epics.PV(trigger_pv, callback=trigger)

    # The following 3 paragraphs where added by chatGPT to speed up pv connection at initialization using a _cache
    # Create all CA PVs up front so they can connect in parallel
    scalar_pvs = {name: get_pv(name) for name in scalars}
    array_pvs  = {name: get_pv(name) for name in arrays}

    # Start metadata PV connections too
    for name in scalars + arrays:
        get_pv(name + ".DESC", auto_monitor=False)
        get_pv(name + ":LongDescription.VAL$", auto_monitor=False)

    # Give all CA PVs a short fixed time to connect in parallel
    t0 = time.time()
    while time.time() - t0 < 0.5:
        epics.ca.poll(evt=0.01)


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

        tsgroup = file.create_group('timestamps') #use optional group for readability
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

            tsgroup.create_dataset(scalar + '.timestamp', (N,), dtype=float)

        scalars = valid_scalars

        file.create_dataset('epoch', (N,), dtype=float)        # add one for time
        file.create_dataset('dT_this_acquisition', (N,), dtype=float) # one one for dT this acquisition
        
        valid_arrays = []

        for array in arrays:
            pv = array_pvs[array]

            if not pv.connected:
                print(f"WARNING: skipping missing array PV {array}")
                continue

            try:
                array_sample = pv.get(timeout=0.2)
                if array_sample is None or len(array_sample) == 0:
                    raise RuntimeError("None returned")
            except Exception:
                print(f"WARNING: skipping unreadable array PV {array}")
                continue

            valid_arrays.append(array)

            print(pv)   # so we see when it crashes

            # Create datasets
            dset = file.create_dataset(
                array,
                shape=(N, len(array_sample)),
                maxshape=(None, len(array_sample)),
                chunks=(1, len(array_sample)),
                dtype=float
            )

            save_epics_metadata(dset, pv)

            tsgroup.create_dataset(array + '.timestamp', (N,), dtype=float)

        arrays = valid_arrays

        # create empty datasets to store arrays repeatedly N times
        valid_images = []     # keep only image PVs that exist
        for image in images:
            try:
                test = ctx.get(image, timeout=0.5)
                if test is None:
                    print(f"WARNING: skipping missing image PV {image}")
                    continue
                valid_images.append(image)
                file.create_group(image)

            except Exception as e:
                print(f"WARNING: skipping missing image PV {image}: {e}")
                continue

        images = valid_images






    start_time = time.time()    # for total run duration
    shot=0  # shot counter
    print("Waiting for trigger")
    try:
        TrigState=0 # reset trigger
        while shot < N:
            epics.ca.poll(evt=0.01)     # chek for new events
        
            # waiting for trigger
            if TrigState == 1:
                # just triggered
                trigger_time=time.time()
                t0_acquisition=time.perf_counter()
                #os.system('clear')  # clear screen

                time.sleep(WAIT_FOR_SLOW_SCOPES) #to allow slow scope readout
                
                with h5py.File(filename, 'a') as file:
                    tsgroup = file['timestamps']
                    
                    # 1. read scalars and write to hdf
                    t0 = t0_acquisition

                    for scalar in scalars:
                        try:
                            value = scalar_pvs[scalar].get()
                            tstamp = scalar_pvs[scalar].timestamp

                            if value is None:
                                print(f"WARNING: scalar PV returned None: {scalar}")
                                continue

                            file[scalar][shot] = value
                            tsgroup[scalar + '.timestamp'][shot] = tstamp

                            t1 = time.perf_counter()
                            print(f"{shot:>5}/{N-1:<5} {tstamp-trigger_time:>13.1f} {scalar[:40]:<40} {value:<16.3g}, dT={(t1-t0)*1000:.3g} ms")
                            t0 = t1

                        except Exception as e:
                            print(f"WARNING: failed scalar {scalar}: {e}")
                            continue

                    file['epoch'][shot] = time.time()   # also save epoch time

                    
                    # 2. read images and write to hdf
                    for image_name in images:
                        image, timestamp = ReadEpicsImage2(image_name)
                        if image is None:
                            print(f"WARNING: image PV failed: {image_name}")
                            continue

                        dset = file[image_name].create_dataset(f"image {shot}", data=image)
                        dset.attrs['timestamp'] = timestamp

                        t1 = time.perf_counter()
                        print(f"{shot:>5}/{N-1:<5} {timestamp-trigger_time:>13.1f} {image_name[:40]:<40} {str(image.shape):<16}, dT={(t1-t0)*1000:.3g} ms")
                        t0=t1
                    
                    # 3. read arrays and write to hdf; do it last, they take the longest to populate
                    for array in arrays:
                        try:
                            vector = array_pvs[array].get()
                            tstamp = array_pvs[array].timestamp

                            if vector is None:
                                print(f"WARNING: array PV returned None: {array}")
                                continue

                            file[array][shot, :] = vector
                            tsgroup[array + '.timestamp'][shot] = tstamp

                            t1 = time.perf_counter()
                            print(f"{shot:>5}/{N-1:<5} {tstamp-trigger_time:>13.1f} {array[:40]:<40} {str(vector.shape):<16}, dT={(t1-t0)*1000:.3g} ms")
                            t0 = t1

                        except Exception as e:
                            print(f"WARNING: failed array {array}: {e}")
                            continue

                    file['dT_this_acquisition'][shot] = time.time()-trigger_time    # also save dT This acquisition
                        
                
                shot+=1
                TrigState = 0   #reset
                print(f"\033[1;31mdT this acquisition: {(time.time()-trigger_time):.3g} s  \033[0m")
        
        print('_' * 77)
        print(f"\033[1;32mRun {filename} complete. Runtime {(time.time()-start_time)/60:.3g} minutes.\033[0m")
        print()

    except KeyboardInterrupt:
        print('program terminated')

    finally:
        ctx.close()
