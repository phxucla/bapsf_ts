"""
TS analysis script modified from Chris's code.
Date: 2025-06-30
Author: J.Han using claude-4-sonnet
==================================================================================
                        THOMSON SCATTERING SPECTRUM ANALYSIS
==================================================================================

This script analyzes Thomson scattering spectral data from HDF5 files created by
the LAPD Thomson scattering diagnostic system. It supports three types of data:

1. SINGLE DELAY DATA (from acquire_Nshots.py)
   - Fixed delay time, multiple shots for statistics
   - Calculates plasma parameters (Te, ne) with error bars

2. DELAY SCAN DATA (from scan_delay.py) 
   - Multiple delay times to study plasma evolution
   - Shows Te and ne vs time with full plasma parameter analysis
   
3. BEAM POSITION SCAN DATA (from scan_beam.py)
   - Multiple beam positions to find optimal beam-plasma overlap
   - Shows signal amplitude vs position for beam alignment

==================================================================================
                                MAIN FUNCTIONS
==================================================================================

CORE ANALYSIS FUNCTION:
-----------------------
analyze_spectrum(filename, delay_value=None, position_value=None, debug=False)
    - Low-level function that processes spectral data
    - Handles background subtraction, wavelength calibration, Gaussian fitting
    - Calculates plasma parameters (Te, ne) from spectral line broadening
    - Used by all higher-level functions

HIGH-LEVEL PROCESSING FUNCTIONS:
--------------------------------
process_single_delay_data(filename, debug=False)
    - For acquire_Nshots.py data files
    - Shows detailed 4-panel analysis plot
    - Reports Te, ne with error bars from multiple shots

process_scan_delay_data(filename, debug=False)  
    - For scan_delay.py data files
    - Analyzes each delay time separately
    - Creates Te vs time and ne vs time plots
    - Shows evolution of plasma parameters

process_scan_beam_data(filename, debug=False)
    - For scan_beam.py data files  
    - Shows Gaussian fits for each beam position (4 panels per figure)
    - Identifies optimal position based on signal amplitude
    - Used for beam alignment optimization

==================================================================================
                            DATA FORMAT REQUIREMENTS
==================================================================================

HDF5 file structure expected:
- Images: "13PICAM1:Pva1:Image/image N" (N = shot number)
- Shot pattern: even shots = background, odd shots = signal
- Background subtraction: signal - background for each pair

For scan_delay.py files:
- "actionlist/BNC3:chB:DelayDesired" contains delay values

For scan_beam.py files:  
- "actionlist/Motor12:PositionInput" contains position values

==================================================================================
                                USAGE EXAMPLES
==================================================================================

# Automatic detection and processing:
python spectrum.py  # (edit filename in main section)

# Manual function calls:
from spectrum import process_single_delay_data, process_scan_delay_data, process_scan_beam_data

# Single delay analysis with debug plots
process_single_delay_data("acquire_data.h5", debug=True)

# Delay scan with time evolution plots  
process_scan_delay_data("delay_scan.h5", debug=True)

# Beam position scan for alignment
process_scan_beam_data("beam_scan.h5", debug=False)

==================================================================================
                            CALIBRATION CONSTANTS
==================================================================================

Wavelength calibration: λ = pixel * 19.80636/511 + 522.918 nm
Temperature calibration: Te = 0.903 * σ² eV (where σ is Gaussian width)
Density calibration: ne = 2.98e8 * Area / 1e13 cm⁻³

==================================================================================
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Enable interactive mode for matplotlib
plt.ion()
plt.rcParams['font.size'] = 12

def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude*np.exp(-(x - mean)**2 / (2*stddev**2))+offset

def analyze_spectrum(filename, delay_value=None, position_value=None, debug=False, debug_mode="full"):
    """
    Analyze Thomson scattering spectrum from HDF5 file.
    
    Parameters:
    -----------
    filename : str
        Path to HDF5 file
    delay_value : float, optional
        Specific delay value to analyze (for scan_delay data)
    position_value : float, optional
        Specific position value to analyze (for scan_beam data)
    debug : bool, optional
        If True, show debug plots or return debug data
    debug_mode : str, optional
        "full" - show all 4 panels (default)
        "gaussian_only" - return Gaussian plot data without showing plots
        
    Returns:
    --------
    dict : Analysis results containing Te, Tmax, Tmin, area, ne
           If debug_mode="gaussian_only", also includes plot data
    """
    
    file = h5py.File(filename,'r')
    
    # Get number of shots
    dataset = file["13PICAM1:Pva1:Image"]
    total_shots = len(dataset)
    
    # Handle delay-specific analysis for scan_delay data
    if delay_value is not None:
        # Get delay data from actionlist
        try:
            delay_data = np.array(file["actionlist/BNC3:chB:DelayDesired"])
            # Find shots corresponding to this delay
            delay_indices = np.where(np.isclose(delay_data, delay_value, rtol=1e-6))[0]
            if len(delay_indices) == 0:
                print(f"No shots found for delay {delay_value}")
                file.close()
                return None
            shot_range = delay_indices
        except KeyError:
            print("No delay data found in file - treating as single delay dataset")
            shot_range = range(0, total_shots, 2)
    # Handle position-specific analysis for scan_beam data
    elif position_value is not None:
        # Get position data from actionlist
        try:
            position_data = np.array(file["actionlist/Motor12:PositionInput"])
            # Find shots corresponding to this position
            position_indices = np.where(np.isclose(position_data, position_value, rtol=1e-6))[0]
            if len(position_indices) == 0:
                print(f"No shots found for position {position_value}")
                file.close()
                return None
            shot_range = position_indices
        except KeyError:
            print("No position data found in file - treating as single position dataset")
            shot_range = range(0, total_shots, 2)
    else:
        # Use all shots for acquire_Nshots data
        shot_range = range(0, total_shots, 2)
    
    # Process images
    profile = np.zeros(512, dtype=float)
    average_profile = np.zeros(512, dtype=float)
    valid_pairs = 0
    
    for i in range(0, len(shot_range), 2):
        if i+1 >= len(shot_range):
            break
            
        n_bg = shot_range[i]
        n_signal = shot_range[i+1]
        
        # Read background and signal
        bg = np.asarray(file.get(f"/13PICAM1:Pva1:Image/image {n_bg}")).astype(float)
        image = np.asarray(file.get(f"/13PICAM1:Pva1:Image/image {n_signal}")).astype(float)
        
        # Subtract background
        image -= bg
        
        # Sum vertically to get horizontal profile
        for xx in range(512):
            profile[xx] = np.sum(image[:, xx])
        
        average_profile += profile
        valid_pairs += 1
    
    file.close()
    
    if valid_pairs == 0:
        print("No valid shot pairs found")
        return None
    
    # Normalize
    average_profile /= valid_pairs
    average_profile *= (-1)  # flip sign
    
    # Wavelength calibration
    wavelength = np.arange(512) * 19.80636 / 511 + 522.918
    
    # Create binned profiles
    binned_wavelength = np.arange(256) * 0.019 * 4 + 523.12
    superbinned_wavelength = np.arange(128) * 0.019 * 8 + 523.0
    supersuperbinned_wavelength = np.arange(64) * 0.019 * 16 + 523.2
    
    binned_profile = np.zeros(256, dtype=float)
    for i in range(256):
        binned_profile[i] = np.sum(average_profile[i*2:i*2+2])
    
    superbinned_profile = np.zeros(128, dtype=float)
    for i in range(128):
        superbinned_profile[i] = np.sum(binned_profile[i*2:i*2+2])
    
    supersuperbinned_profile = np.zeros(64, dtype=float)
    for i in range(64):
        supersuperbinned_profile[i] = np.sum(superbinned_profile[i*2:i*2+2])
    
    # Fit Gaussian to supersuperbinned data
    mask = np.ones(64, dtype=bool)
    mask[27:33] = False  # Exclude central region
    
    try:
        popt, cov = optimize.curve_fit(
            gaussian, 
            supersuperbinned_wavelength[mask], 
            supersuperbinned_profile[mask], 
            p0=[800, 532, 6, 0]
        )
        Gauss = gaussian(supersuperbinned_wavelength, *popt)
        
        # Calculate results using offset-corrected Gaussian for density
        Gauss_for_density = Gauss - np.mean(Gauss[0:10])  # separate copy for density calculation
        sigmaerror = np.sqrt(cov[2][2])
        fwhm = popt[2] * 2.355
        Te = 0.903 * popt[2]**2
        Tmax = 0.903 * (popt[2] + sigmaerror)**2
        Tmin = 0.903 * (popt[2] - sigmaerror)**2
        area = np.sum(Gauss_for_density)  # use offset-corrected version
        ne = 2.98e8 * np.sum(Gauss_for_density) / 1e13  # use offset-corrected version
        
        results = {
            'Te': Te,
            'Tmax': Tmax, 
            'Tmin': Tmin,
            'area': area,
            'ne': ne,
            'sigma': popt[2],
            'sigma_error': sigmaerror,
            'fwhm': fwhm,
            'valid_pairs': valid_pairs,
            'delay': delay_value,
            'position': position_value
        }
        print(f"\nResults:")
        print(f"Te = {results['Te']:.2f} eV")
        print(f"Tmax = {results['Tmax']:.2f} eV") 
        print(f"Tmin = {results['Tmin']:.2f} eV")
        print(f"ne = {results['ne']:.2f} ×10¹³ cm⁻³")
        print(f"Area = {results['area']:.0f}")
        
        # Add plot data for gaussian_only mode
        if debug and debug_mode == "gaussian_only":
            results['plot_data'] = {
                'wavelength': supersuperbinned_wavelength,
                'data': supersuperbinned_profile,
                'fit': Gauss,
                'delay': delay_value,
                'position': position_value
            }
        
        # Debug plotting for full mode
        if debug and debug_mode == "full":
            fig, axes = plt.subplots(2, 2, dpi=300, 
                                   gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
            
            # Create appropriate title string
            title_str = ""
            if delay_value is not None:
                title_str = f" (delay={delay_value}s)"
            elif position_value is not None:
                title_str = f" (position={position_value:.3f}cm)"
            
            fig.canvas.manager.set_window_title(f'Spectrum Analysis - {filename}{title_str}')
            
            # Left column: Binned profiles with shared x-axis
            # Plot 1: Super-binned profile (4x)
            axes[0, 0].plot(superbinned_wavelength, superbinned_profile)
            axes[0, 0].set_ylabel('Intensity')
            axes[0, 0].set_title('Super-binned Profile (4x)')
            axes[0, 0].tick_params(labelbottom=False)  # Remove x-tick labels
            
            # Plot 2: Super-super-binned profile (8x) - shares x with above
            axes[1, 0].plot(supersuperbinned_wavelength, supersuperbinned_profile)
            axes[1, 0].set_xlabel('Wavelength (nm)')
            axes[1, 0].set_ylabel('Intensity')
            axes[1, 0].set_title('Super-super-binned Profile (8x)')

            
            # Right column: Spectra with shared x and y axis
            # Plot 3: Original spectrum
            axes[0, 1].plot(wavelength, average_profile)
            axes[0, 1].set_title('Original Spectrum')
            axes[0, 1].tick_params(labelbottom=False)  # Remove x ticks labels
            
            # Plot 4: Final spectrum with Gaussian fit - shares x and y with above
            axes[1, 1].plot(supersuperbinned_wavelength, supersuperbinned_profile, 'b-', label='Data')
            axes[1, 1].plot(supersuperbinned_wavelength, Gauss, 'r-', linewidth=2, label='Gaussian Fit')
            axes[1, 1].set_xlabel('Wavelength (nm)')
            axes[1, 1].tick_params(labelleft=False)  # Remove y-tick labels
            axes[1, 1].set_title('Final Spectrum with Gaussian Fit')
            
            # Share y-axis for right column
            axes[0, 1].sharey(axes[1, 1])
            
            plt.draw()
            plt.pause(0.001)
            
        return results
        
    except Exception as e:
        print(f"Error in Gaussian fitting: {e}")
        return None

def analyze_scan_delay(ifn, debug=False, debug_mode="full"):
    """
    Analyze scan_delay data for all delay values.
    
    Parameters:
    -----------
    ifn : str
        Path to HDF5 file from scan_delay.py
    debug : bool, optional
        If True, show debug plots for each delay
    debug_mode : str, optional
        "full" or "gaussian_only" - passed to analyze_spectrum
        
    Returns:
    --------
    dict : Results for all delays
    """
    
    # Get unique delay values
    with h5py.File(ifn, 'r') as file:
        try:
            delay_data = np.array(file["actionlist/BNC3:chB:DelayDesired"])
            unique_delays = np.unique(delay_data)
        except KeyError:
            print("No delay data found - treating as single delay dataset")
            return {"single": analyze_spectrum(ifn, debug=debug, debug_mode=debug_mode)}
    
    print(f"Found {len(unique_delays)} unique delays: {unique_delays}")
    
    # Analyze each delay
    results = {}
    for delay in unique_delays:
        print(f"\nAnalyzing delay {delay*1000:.1f} ms...")
        result = analyze_spectrum(ifn, delay_value=delay, debug=debug, debug_mode=debug_mode)
        if result is not None:
            results[delay] = result
            print(f"  Te = {result['Te']:.2f} eV, ne = {result['ne']:.2f} ×10¹³ cm⁻³")
    
    return results

def process_scan_delay_data(ifn, debug=False):
    """
    Complete processing of scan_delay data including analysis and plotting.
    
    Parameters:
    -----------
    ifn : str
        Path to HDF5 file from scan_delay.py
    debug : bool, optional
        If True, show Gaussian fit plots for each delay in a single figure
    """
    print("Detected scan_delay data")
    
    # Use gaussian_only mode for debug to collect plot data
    debug_mode = "gaussian_only" if debug else "full"
    results = analyze_scan_delay(ifn, debug=debug, debug_mode=debug_mode)
    
    # Create debug plot with all Gaussian fits if debug=True
    if debug and len(results) > 1:
        n_delays = len(results)
        cols = min(3, n_delays)  # Max 3 columns
        rows = (n_delays + cols - 1) // cols  # Calculate rows needed
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), dpi=300)
        fig.suptitle(f'Gaussian Fits for All Delays - {ifn}')
        
        # Handle single row case
        if rows == 1:
            axes = axes if n_delays > 1 else [axes]
        
        plot_idx = 0
        for delay, result in results.items():
            if 'plot_data' in result:
                row = plot_idx // cols
                col = plot_idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                
                plot_data = result['plot_data']
                ax.plot(plot_data['wavelength'], plot_data['data'], 'b-', label='Data')
                ax.plot(plot_data['wavelength'], plot_data['fit'], 'r-', linewidth=2, label='Gaussian Fit')
                ax.set_title(f'Delay {delay*1000:.1f} ms\nTe={result["Te"]:.1f} eV')
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Intensity')
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.show(block=False)
    
    # Plot results vs delay
    if len(results) > 1:
        delays = []
        Te_values = []
        Tmax_values = []
        Tmin_values = []
        ne_values = []
        
        for delay, result in results.items():
            delays.append(delay * 1000)  # Convert to ms
            Te_values.append(result['Te'])
            Tmax_values.append(result['Tmax'])
            Tmin_values.append(result['Tmin'])
            ne_values.append(result['ne'])
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(2, 1, dpi=300)
        
        # Temperature plot
        ax1.errorbar(delays, Te_values, 
                   yerr=[np.array(Te_values) - np.array(Tmin_values),
                         np.array(Tmax_values) - np.array(Te_values)],
                   fmt='o-', capsize=5, capthick=2, markersize=8)
        ax1.set_xlabel('Delay (ms)')
        ax1.set_ylabel('Electron Temperature (eV)')
        ax1.set_title('Electron Temperature vs Delay')
        ax1.grid(True, alpha=0.3)
        
        # Density plot
        ax2.plot(delays, ne_values, 'o-', markersize=8, color='red')
        ax2.set_xlabel('Delay (ms)')
        ax2.set_ylabel('Electron Density (×10¹³ cm⁻³)')
        ax2.set_title('Electron Density vs Delay')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        plt.show(block=True)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY RESULTS")
        print("="*60)
        for delay, result in results.items():
            print(f"Delay {delay*1000:5.1f} ms: Te = {result['Te']:5.2f} eV, "
                  f"ne = {result['ne']:5.2f} ×10¹³ cm⁻³")
        print("="*60)

def process_single_delay_data(ifn, debug=False):
    """
    Process single delay data (from acquire_Nshots.py).
    
    Parameters:
    -----------
    ifn : str
        Path to HDF5 file from acquire_Nshots.py
    debug : bool, optional
        If True, show debug plots (full 4-panel layout)
    """
    print("Detected single delay data (acquire_Nshots)")
    result = analyze_spectrum(ifn, debug=debug, debug_mode="full")
    plt.show(block=True)

def analyze_scan_beam(ifn, debug=False, debug_mode="full"):
    """
    Analyze scan_beam data for all beam positions.
    
    Parameters:
    -----------
    ifn : str
        Path to HDF5 file from scan_beam.py
    debug : bool, optional
        If True, show debug plots for each position
    debug_mode : str, optional
        "full" or "gaussian_only" - passed to analyze_spectrum
        
    Returns:
    --------
    dict : Results for all positions
    """
    
    # Get unique position values
    with h5py.File(ifn, 'r') as file:
        try:
            position_data = np.array(file["actionlist/Motor12:PositionInput"])
            unique_positions = np.unique(position_data)
        except KeyError:
            print("No beam position data found - treating as single position dataset")
            return {"single": analyze_spectrum(ifn, debug=debug, debug_mode=debug_mode)}
    
    print(f"Found {len(unique_positions)} unique positions: {unique_positions} cm")
    
    # Analyze each position
    results = {}
    for position in unique_positions:
        print(f"\nAnalyzing position {position:.3f} cm...")
        result = analyze_spectrum(ifn, position_value=position, debug=debug, debug_mode=debug_mode)
        if result is not None:
            results[position] = result
            print(f"  Te = {result['Te']:.2f} eV, ne = {result['ne']:.2f} ×10¹³ cm⁻³")
    
    return results

def process_scan_beam_data(ifn, debug=False):
    """
    Process scan_beam data to find beam position using Gaussian fits.
    Shows 4 panels per figure, creating multiple figures as needed.
    
    Parameters:
    -----------
    ifn : str
        Path to HDF5 file from scan_beam.py
    debug : bool, optional
        Not used, kept for compatibility
    """
    print("Detected scan_beam data - finding beam position")
    
    # Get unique position values and analyze each one
    with h5py.File(ifn, 'r') as file:
        try:
            position_data = np.array(file["actionlist/Motor12:PositionInput"])
            unique_positions = np.unique(position_data)
        except KeyError:
            print("No beam position data found")
            return
    
    print(f"Found {len(unique_positions)} unique positions: {unique_positions} cm")
    
    # Collect plot data for each position
    plot_data = {}
    for position in unique_positions:
        print(f"Processing position {position:.3f} cm...")
        
        # Get the spectral data for this position (simplified analysis)
        with h5py.File(ifn, 'r') as file:
            # Get shots for this position
            position_indices = np.where(np.isclose(position_data, position, rtol=1e-6))[0]
            
            # Process images for this position
            profile = np.zeros(512, dtype=float)
            average_profile = np.zeros(512, dtype=float)
            valid_pairs = 0
            
            for i in range(0, len(position_indices), 2):
                if i+1 >= len(position_indices):
                    break
                    
                n_bg = position_indices[i]
                n_signal = position_indices[i+1]
                
                # Read background and signal
                bg = np.asarray(file.get(f"/13PICAM1:Pva1:Image/image {n_bg}")).astype(float)
                image = np.asarray(file.get(f"/13PICAM1:Pva1:Image/image {n_signal}")).astype(float)
                
                # Subtract background
                image -= bg
                
                # Sum vertically to get horizontal profile
                for xx in range(512):
                    profile[xx] = np.sum(image[:, xx])
                
                average_profile += profile
                valid_pairs += 1
            
            if valid_pairs == 0:
                continue
                
            # Normalize
            average_profile /= valid_pairs
            average_profile *= (-1)  # flip sign
            
            # Create super-super-binned profile for fitting
            supersuperbinned_wavelength = np.arange(64) * 0.019 * 16 + 523.2
            supersuperbinned_profile = np.zeros(64, dtype=float)
            
            # Bin down the data
            binned_profile = np.zeros(256, dtype=float)
            for i in range(256):
                binned_profile[i] = np.sum(average_profile[i*2:i*2+2])
            
            superbinned_profile = np.zeros(128, dtype=float)
            for i in range(128):
                superbinned_profile[i] = np.sum(binned_profile[i*2:i*2+2])
            
            for i in range(64):
                supersuperbinned_profile[i] = np.sum(superbinned_profile[i*2:i*2+2])
            
            # Fit Gaussian
            mask = np.ones(64, dtype=bool)
            mask[27:33] = False  # Exclude central region
            
            try:
                popt, cov = optimize.curve_fit(
                    gaussian, 
                    supersuperbinned_wavelength[mask], 
                    supersuperbinned_profile[mask], 
                    p0=[800, 532, 6, 0]
                )
                Gauss = gaussian(supersuperbinned_wavelength, *popt)
                
                # Store plot data
                plot_data[position] = {
                    'wavelength': supersuperbinned_wavelength,
                    'data': supersuperbinned_profile,
                    'fit': Gauss,
                    'amplitude': popt[0],
                    'center': popt[1],
                    'sigma': popt[2],
                    'offset': popt[3],
                    'valid_pairs': valid_pairs
                }
                
            except Exception as e:
                print(f"Failed to fit Gaussian for position {position:.3f} cm: {e}")
    

    if plot_data:
        print("\n" + "="*50)
        print("BEAM POSITION SCAN RESULTS")
        print("="*50)
        
        # Collect all amplitudes for outlier detection (use absolute values)
        all_amplitudes = [abs(data['amplitude']) for data in plot_data.values()]
        mean_amplitude = np.mean(all_amplitudes)
        std_amplitude = np.std(all_amplitudes)
        
        # Filter outliers
        valid_positions = {}
        eliminated_positions = {}
        
        for position in sorted(plot_data.keys()):
            data = plot_data[position]
            amplitude = data['amplitude']
            abs_amplitude = abs(amplitude)
            
            # Check if absolute amplitude is within n standard deviations
            if abs(abs_amplitude - mean_amplitude) <= 0.55 * std_amplitude:
                valid_positions[position] = data
                status = ""
            else:
                eliminated_positions[position] = data
                status = " [ELIMINATED - Poor fit]"
            
            print(f"Position {position:6.3f} cm: Amplitude = {amplitude:6.0f}, "
                  f"Shots = {data['valid_pairs']:2d}{status}")
        
        print("="*50)
        
        # Find best position from valid positions only (use absolute amplitude)
        if valid_positions:
            best_position = max(valid_positions.keys(), 
                              key=lambda pos: abs(valid_positions[pos]['amplitude']))
            best_amplitude = valid_positions[best_position]['amplitude']
            print(f"BEST POSITION: {best_position:.3f} cm (Amplitude: {best_amplitude:.0f})")
            
            if eliminated_positions:
                print(f"NOTE: {len(eliminated_positions)} position(s) eliminated due to poor Gaussian fits")
                eliminated_list = [f"{pos:.3f}" for pos in eliminated_positions.keys()]
                print(f"Eliminated positions: {', '.join(eliminated_list)} cm")
        else:
            print("NO VALID POSITIONS FOUND - All fits were poor")
        
        print("="*50)
        print()  # Add blank line before plotting message
    
    # Create plots with 4 panels per figure AFTER printing results
    if plot_data:
        positions = sorted(plot_data.keys())
        n_positions = len(positions)
        panels_per_figure = 4
        n_figures = (n_positions + panels_per_figure - 1) // panels_per_figure

        figures = []
        for fig_idx in range(n_figures):
            # Create 2x2 subplot layout for each figure
            fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=300)
            axes = axes.flatten()  # Convert 2x2 to 1D array for easier indexing
            
            # Calculate which positions go in this figure
            start_pos = fig_idx * panels_per_figure
            end_pos = min(start_pos + panels_per_figure, n_positions)
            positions_in_fig = positions[start_pos:end_pos]
            
            # Plot data for positions in this figure
            for panel_idx, position in enumerate(positions_in_fig):
                data = plot_data[position]
                ax = axes[panel_idx]
                
                # Plot data and fit
                ax.plot(data['wavelength'], data['data'], 'b-', alpha=0.7, label='Data')
                ax.plot(data['wavelength'], data['fit'], 'r-', linewidth=2, label='Gaussian')
                ax.set_title(position)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)
                
                # Set consistent y-scale
                ax.set_ylim(bottom=min(0, np.min(data['data'])*1.1))
            
            # Hide unused panels in this figure
            for panel_idx in range(len(positions_in_fig), panels_per_figure):
                axes[panel_idx].axis('off')
            
            plt.tight_layout()
            figures.append(fig)
        
        # Show all figures
        for fig in figures:
            plt.figure(fig.number)
            plt.show(block=False)
        
        # Block on the last figure to keep all open
        if figures:
            plt.show(block=True)

if __name__ == "__main__":
    # Configuration
    ifn = r"D:\data\LAPD\TS\run29-beamScan-10rep-delay19p5ms-2025-06-30.h5"
    
    # Check if this is scan_delay data, scan_beam data, or single delay data
    try:
        with h5py.File(ifn, 'r') as file:
            if "actionlist/BNC3:chB:DelayDesired" in file:
                process_scan_delay_data(ifn, debug=True)
            elif "actionlist/Motor12:PositionInput" in file:
                process_scan_beam_data(ifn, debug=True)
            else:
                process_single_delay_data(ifn, debug=True)
                    
    except FileNotFoundError:
        print(f"File {ifn} not found")
    except Exception as e:
        print(f"Error: {e}")