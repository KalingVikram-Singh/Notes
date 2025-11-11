import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import tifffile  # Library for reading multi-page TIFF files
from scipy.optimize import curve_fit

# --- 1. Data Loading ---
def load_spad_data(filepath):
    try:
        dataset = tifffile.imread(filepath)
        print(f"Dataset shape: {dataset.shape} (frames, rows, cols)")
        
        return dataset.astype(np.float64)
    except FileNotFoundError:
        return None

def divide_by_16(dataset):
    divided_data = dataset / 16.0
    return divided_data

def mean_image(dataset):
    return np.mean(dataset, axis=0) 

def mean_area(dataset, x_start, x_end, y_start, y_end):
    area = dataset[:, y_start:y_end, x_start:x_end]
    mean_area_values = np.mean(area, axis=(1, 2))
    average_counts = np.mean(mean_area_values)
    return average_counts

def std_image(dataset):
    return np.std(dataset, axis=0)

def std_area(dataset, x_start, x_end, y_start, y_end):
    area = dataset[:, y_start:y_end, x_start:x_end]
    std_area_values = np.std(area, axis=(1, 2))
    std_counts = np.mean(std_area_values)
    return std_counts

def std_histo(dataset, x_start, x_end, y_start, y_end, bins=50):
    area = dataset[:, y_start:y_end, x_start:x_end]
    std_area_values = np.std(area, axis=(1, 2))
    plt.figure()
    plt.hist(std_area_values, bins=bins, density=False, alpha=0.7, color='blue')
    plt.title('Histogram of Standard Deviation in Selected Area')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Number of Frames')
    plt.grid(True)
    plt.show()
    return std_area_values


def plot_mean(dataset,title):
    mean_img = mean_image(dataset)
    plt.figure()
    plt.imshow(mean_img, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Mean Counts')
    plt.title(title)
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    plt.show()
    return mean_img

def plot_std(dataset,title):
    std_dev = std_image(dataset)
    plt.figure()
    plt.imshow(std_dev, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Standard Deviation')
    plt.title(title)
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    plt.show()
    return std_dev

def plot_histogram(data, title, xlabel, ylabel, bins=100):
    plt.figure()
    plt.hist(data.flatten(), bins=bins, density=False, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    FILE_PATH = r"C:\Users\kvs4000\Desktop\1s_dark_GS650.tif"
    dataset = load_spad_data(FILE_PATH)
    new_data_dark = dataset/16.0

    # FILE_PATH = r"C:\Users\kvs4000\Desktop\Bright_GS650.tif"
    # dataset = load_spad_data(FILE_PATH)
    # new_data_bright = dataset/16.0

    FILE_PATH = r"C:\Users\kvs4000\Desktop\dark_GS650_10ms.tif"
    dataset = load_spad_data(FILE_PATH)
    new_data_dark_10ms = dataset/16.0 

    gain = 1.57 #Calculated from previous PTC analysis
    difference = -(mean_area(new_data_dark_10ms,100, 120, 100, 120) - mean_area(new_data_dark, 100, 120, 100, 120))
    print('Mean of the area gives the offset (1s exposure ):',mean_area(new_data_dark, 100, 120, 100, 120))
    print('Mean of the area gives the offset (10 ms exposure ):',mean_area(new_data_dark_10ms, 100, 120, 100, 120))
    print('Difference in mean between 1s and 10ms exposure  (dark current per pixel per 0.99s):',difference)
    print('Estimated Dark Current (ADU/pixel/s):', difference / 0.99)
    print('Estimated Dark Current (e/pixel/s):', (difference/gain))
    print('Standard Deviation of the area gives the Gain x ( Dark noise + RN) for 1s dark exposure:',std_area(new_data_dark, 100, 120, 100, 120))
    print('Offset ADU from 1s dark exposure:', mean_area(new_data_dark, 100, 120, 100, 120) - difference, 'ADU')



    # plot_histogram(new_data_dark, 'Histogram of Counts (dark Current - 1s)', 'Counts', 'Number of pixels', bins=200)
    # plot_mean(new_data_dark,'Mean Image (dark Current - 1s)')
    # plot_std(new_data_dark,'Standard Deviation Image (dark Current - 1s)')
    # print('Mean of the area gives the offset (1s exposure ):',mean_area(new_data_dark, 100, 120, 100, 120))
    # print('STD of the area gives the Gain x Dark current levels (1s exposure ):',std_area(new_data_dark, 100, 120, 100, 120))

    # plot_histogram(new_data_dark_10ms, 'Histogram of Counts (dark - 10ms)', 'Counts', 'Number of pixels', bins=200)
    # plot_mean(new_data_dark_10ms,'Mean Image (dark - 10ms)')   
    # plot_std(new_data_dark_10ms,'Standard Deviation Image (dark - 10ms)')
    # print('Mean of the area gives the offset: (10 ms exposure)',mean_area(new_data_dark_10ms, 232, 242, 20, 120))
    # print('STD of the area gives the Gain x Dark current levels (10 ms exposure):',std_area(new_data_dark_10ms, 232, 242, 20, 120))
    
    # difference = -(mean_area(new_data_dark_10ms,100, 120, 100, 120) - mean_area(new_data_dark, 100, 120, 100, 120))
    # print('Difference in mean between 1s and 10ms exposure (should relate to shot noise):', difference)
    # print('Estimated Gain from difference:', difference / 0.99)  # Assuming 0.99 is the expected photon count difference
    # print('Estimated Dark Current from 1s exposure:', mean_area(new_data_dark, 100, 120, 100, 120) - (difference / 0.99))
    # print('Estimated Dark Current from 10ms exposure:', (mean_area(new_data_dark_10ms, 100, 120, 100, 120) - (difference / 0.99)) * 0.01)
    

    # plot_histogram(new_data_bright, 'Histogram of Counts (bright - 5ms)', 'Counts', 'Number of pixels', bins=200)
    # plot_mean(new_data_bright,'Mean Image (bright - 5ms)')   
    # plot_std(new_data_bright,'Standard Deviation Image (bright - 5ms)')
    print("Done")
    print("Analysis completed.")