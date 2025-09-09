import numpy as np
from brian2 import TimedArray, ms, second, Hz

def generate_oscillatory_input(freq, duration, noise=0.1):
    t = np.arange(0, duration/ms, 1) * ms
    pure = np.sin(2 * np.pi * freq * t)
    noisy = pure + noise * np.random.randn(len(pure))
    return TimedArray(np.clip(noisy, 0, None), dt=1 * ms)

def generate_stimulus_sequence(duration, n_items=10):
    stimulus_sequence = np.random.randint(2, size=n_items)
    return TimedArray(stimulus_sequence, dt=duration / len(stimulus_sequence))
