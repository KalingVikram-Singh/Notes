import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tifffile as tiff
from matplotlib.widgets import RectangleSelector
import sys
from typing import List, Tuple, Optional

def loadimage(filepath: str) -> Optional[np.ndarray]:
    print(f"Filepath: {filepath}")
    stack = tiff.imread(filepath)        
    print(f"Loaded stack shape: {stack.shape}")
    print(f"Frames: {stack.shape[0]}, Size: {stack.shape[1]}x{stack.shape[2]}.")
    return stack


def select_rois_interactively(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    rois = []
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        x_start, x_end = min(x1, x2), max(x1, x2)
        y_start, y_end = min(y1, y2), max(y1, y2)
        
        roi = (y_start, y_end, x_start, x_end)
        rois.append(roi)
        print(f"ROI #{len(rois)} selected: {roi}")

    fig, ax = plt.subplots(figsize=(12, 9))
    
    ax.imshow(image, cmap='gray', origin='upper', aspect='equal')
    ax.set_title("Select >2 bright regions. Close window to continue.")

    print("1. Click and drag on the image to select a rectangular region.")
    selector = RectangleSelector(ax, onselect, useblit=True,
                                 button=[1],  # Left mouse button
                                 minspanx=5, minspany=5,
                                 spancoords='pixels',
                                 interactive=True)
    
    plt.show()  
    return rois

def analyze_ccd_data(
    ccd_stack: np.ndarray, 
    rois: List[Tuple[int, int, int, int]],
    manual_offset: float  
):
    num_frames, height, width = ccd_stack.shape
    print(f"\nAnalyzing stack with shape: ({num_frames}, {height}, {width})")
    # We are only using the manually provided offset (which is 0 for now)

    offset_adu = manual_offset

    #--- Photon Transfer curve part ---
    print("--- Step 2: Photon Transfer Curve ---")
    mean_signals = []
    variances = []
    # Loop through ALL selected ROIs, treating them all as data points
    for i, (y_s, y_e, x_s, x_e) in enumerate(rois):
        region_data = ccd_stack[:, y_s:y_e, x_s:x_e]
        temporal_mean_per_pixel = np.mean(region_data, axis=0)
        mean_signal_for_region = np.mean(temporal_mean_per_pixel)
        temporal_var_per_pixel = np.var(region_data, axis=0)
        mean_variance_for_region = np.mean(temporal_var_per_pixel)
        # Subtract the manual offset ( we took this to be 0 in the analysis, it doesnt affect the gain slope)
        mean_signals.append(mean_signal_for_region - offset_adu)
        variances.append(mean_variance_for_region)
        # Print info for every ROI
        print(f"For region {i},  Mean Signal (net) = {mean_signals[-1]:.2f} counts, Variance = {variances[-1]:.2f} ADU^2")
    mean_signals = np.array(mean_signals)
    variances = np.array(variances)

    slope, intercept = np.polyfit(mean_signals, variances, 1)

    gain = slope
    rn_from_intercept_adu = np.sqrt(max(intercept, 0))

    print(f"\nPTC Fit Slope (Gain): {gain:.4f} ADU/e-")
    print(f"Read Noise from PTC Intercept: {rn_from_intercept_adu:.2f} ADU")
    print("-------------------------------------\n")

    # --- Final Results ---------
    print(f"Gain: {gain:.2f} ADU/e-")

    # ---------------------Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all points as data
    ax.scatter(mean_signals, variances, label='Data from ROIs', zorder=5, alpha=0.8)
    fit_x_start = min(0, mean_signals.min())
    fit_x = np.linspace(fit_x_start, mean_signals.max(), 100)
    fit_y = gain * fit_x + intercept
    ax.plot(fit_x, fit_y, 'r--', label=f'Linear Fit (Gain = {gain:.2f} ADU/e-)', linewidth=2)  
    ax.set_title('Photon Transfer Curve (PTC)', fontsize=16)
    ax.set_xlabel('Mean Signal (ADU, offset not substracted)', fontsize=12)
    ax.set_ylabel('Variance (ADU$^2$)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5) 
    plt.tight_layout()
    plt.show()

def main():
    tiff_file_path = r"C:\Users\kvs4000\Desktop\Bright_GS650.tif"
    ccd_data = loadimage(tiff_file_path)
    if ccd_data is None:
        print("Exiting analysis.")
        sys.exit(1)
    ccd_data = (ccd_data) / 16.0

    print("\nMerging all frames into a single image for ROI selection...")
    mean_image = np.mean(ccd_data, axis=0)
    rois_from_user = select_rois_interactively(mean_image)
    if rois_from_user and len(rois_from_user) >= 2:
        manual_offset: Optional[float] = None
        while manual_offset is None:
            try:
                offset_input = input("\nEnter manual offset (ADU): ")
                if offset_input.strip():  # Check input is not empty
                    manual_offset = float(offset_input)
                else:
                    print("Input cannot be empty. Please enter a number.")
            except ValueError:
                print("Invalid input. Please enter a valid number (e.g., 100.5)")
        print(f"\n{len(rois_from_user)} ROIs selected.")

        analyze_ccd_data(ccd_data, rois_from_user, manual_offset=manual_offset)     
    else:
        print("\nNot enough ROIs were selected.")
        print("Minimum 2 bright regions required.")
if __name__ == '__main__':
    main()