import numpy as np
from scipy.signal import welch, coherence, butter, filtfilt, hilbert, find_peaks
import logging
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

def analyze_lfp_bands(lfp_data, fs, win_seconds=2):
    win_size = int(fs * win_seconds)
    if len(lfp_data) < win_size:
        logger.warning("LFP data is too short for the requested window size. Skipping analysis.")
        return None
    
    try:
        freqs, psd = welch(lfp_data, fs, nperseg=win_size)

        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        df = freqs[1] - freqs[0]
        band_powers = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[mask]) * df
            band_powers[band_name] = max(band_power, 1e-10)
        
        return freqs, psd, band_powers
    except Exception as e:
        logger.error(f"Error during Welch PSD calculation: {e}")
        return None

def analyze_isi_distribution(spike_monitor):
    all_isis = []
    for _, spikes in spike_monitor.items():
        if len(spikes) > 1:
            isis = np.diff(spikes)
            all_isis.extend(isis)
    all_isis = np.array(all_isis)
    all_isis = all_isis[all_isis < 1.0]
    return all_isis

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def compare_isi_distributions(isis1, isis2): 
    if isis1 is None or isis2 is None or len(isis1) == 0 or len(isis2) == 0:
        return None, None
    try:
        ks_statistic, p_value = ks_2samp(isis1, isis2)
        return ks_statistic, p_value
    except Exception as e:
        logger.error(f"Error during KS test: {e}")
        return None, None

def calculate_lfp(state_monitor): 
    if 'I_syn' not in state_monitor.variables:
        logger.error("StateMonitor must record 'I_syn' to calculate LFP.")
        return None 
    lfp = np.mean(state_monitor.I_syn / 0.001, axis=0)
    return lfp

def compute_coherence(signal1, signal2, fs=1000):
    nperseg = min(1024, len(signal1))
    f, Cxy = coherence(signal1, signal2, fs=fs, nperseg=nperseg)
    return f, Cxy

def infer_cognitive_state(theta_power, gamma_power, mean_firing_rate):
    if theta_power is None or gamma_power is None:
        return "Unknown"
    if theta_power > gamma_power and mean_firing_rate > 10:
        cognitive_state = "Focused"
    elif mean_firing_rate < 2:
        cognitive_state = "Resting"
    else:
        cognitive_state = "Distracted"
    return cognitive_state
