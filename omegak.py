import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, interpolate
from scipy.fftpack import fft, ifft
from PIL import Image

def process_gbsar_data(filename, step_size=0.01, gbsar_direction="L", tsweep=155e-3, bw=1300e6, fc=24e9, Rs=0.2, zpad=256, zpad2=256):
    if type(filename) == str:
        data = np.rot90(np.load(filename), 3)
    elif type(filename) == np.ndarray:
        data = np.rot90(filename, 3)
    else:
        if filename.size != (1024, 60):
            filename = filename.resize((60,1024))
        filename = np.array(filename).astype(np.float32)/255.
        #filename = np.flip(filename, 1)
        data = np.rot90(np.array(filename), 3)

    c = 3e8  # speed of light [m/s]
    gamma = bw / tsweep  # chirp rate [Hz/s]
    sweep_samples = len(data[0])  # number of frequency samples in chirp
    num_of_steps = len(data)  # number of GBSAR steps (in stop-and-go mode)
    fs = int(sweep_samples / tsweep)  # sampling frequency [Hz]
    
    # Flip data if needed
    if gbsar_direction == "L":
        data = np.flip(data, 0)
    
    aperture_length = step_size*num_of_steps
    print(" ")
    print("** GBSAR parameters **")
    print("Step size: ", step_size*1e2, " cm")
    print("Number of steps: ", num_of_steps)
    print("Total aperture: ", aperture_length, " m")
    print("Number of chirp samples: ", sweep_samples)
    print("Central frequency: ", fc/1e9, " GHz")
    print("Bandwidth: ", bw/1e6, "MHz")
    
    # Linear mask for low frequencies
    num_of_steps_for_mask = 3
    mask = np.ones(sweep_samples)
    mask[:num_of_steps_for_mask] = np.linspace(0, 1, num_of_steps_for_mask)
    mask[-num_of_steps_for_mask:] = np.linspace(1, 0, num_of_steps_for_mask)
    for index, one_step in enumerate(data):
        fmcw_fft = fft(signal.detrend(one_step))
        remove_low_freq = fmcw_fft * mask
        data[index] = ifft(remove_low_freq)
    
    # Hilbert transform, Residual Video Phase compensation, and Hanning window
    def hilbert_rvp(x, fs, gamma):
        y = np.fft.fft(x, axis=-1)
        y[:, :int(sweep_samples / 2) + 1] = 0  # Zero positive frequencies
        f = np.linspace(-fs / 2, fs / 2, y.shape[1])
        y *= np.exp(1j * np.pi * f**2 / gamma)
        return np.fft.ifft(y, axis=-1)
    
    data_hilbert = hilbert_rvp(data, fs, gamma)
    data_hanning = [i * np.hanning(sweep_samples) for i in data_hilbert]
    
    # Zero padding and FFT
    data_zpad = [np.zeros(sweep_samples)] * zpad
    data_zpad[int(zpad / 2) - int(num_of_steps / 2):int(zpad / 2) + int(num_of_steps / 2)] = data_hanning
    data_fft = np.fft.fft(data_zpad, axis=0)
    data_fft = np.fft.fftshift(data_fft, axes=0)
    
    # Referent Function Multiplication
    kx = np.linspace(-np.pi / step_size, np.pi / step_size, len(data_fft))
    kr = np.linspace(((4 * np.pi / c) * (fc - bw / 2)), ((4 * np.pi / c) * (fc + bw / 2)), sweep_samples)
    
    phi_mf = np.zeros((len(data_fft), sweep_samples))
    for index_j, j in enumerate(data_fft):
        for index_i, i in enumerate(j):
            phi_mf[index_j, index_i] = -Rs * kr[index_i] + Rs * (kr[index_i]**2 - kx[index_j]**2)**0.5
    smf = np.e**(1j * phi_mf)
    
    S_mf = data_fft * smf
    
    # Stolt transform
    Ky_even = kr
    ky = []
    S_st = []
    
    for count in range(len(kx)):
        ky.append((kr**2 - kx[count]**2)**0.5)
        func = interpolate.interp1d(ky[count], S_mf[count], fill_value="extrapolate")
        S_st.append(func(Ky_even))
    
    S_st_hanning = [i * np.hanning(len(S_st[0])) for i in S_st]
    
    # Zero padding and 2D IFFT
    S_st = np.pad(S_st_hanning, ((zpad2, zpad2), (0, 0)), mode='constant')
    
    data_final_ifft_range = np.fft.ifft(S_st, axis=1)
    data_final_ifft_range = np.fft.fftshift(data_final_ifft_range, axes=1)
    data_final_ifft_range = np.fft.fftshift(data_final_ifft_range, axes=0)
    data_final_ifft_range = np.fft.ifft(data_final_ifft_range, axis=0)
    
    final_radar_image = np.asmatrix(data_final_ifft_range)
    final_radar_image = np.rot90(final_radar_image)
    final_radar_image = final_radar_image[480:520, 300:480]
    
    # Visualization
    #visualize(final_radar_image)
    
    return final_radar_image


def visualize(image):
    plt.figure()
    sns.heatmap(np.abs(image), cbar=False)
    plt.title("Final Radar Image")
    plt.show()