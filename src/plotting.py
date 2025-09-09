import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, Hz, mV

def plot_comparison(plot_data):
    real_mean_rate = plot_data['real_mean_rate']
    real_std_rate = plot_data['real_std_rate']
    sim_mean_rate = plot_data['sim_mean_rate']
    real_isis = plot_data['real_isis']
    sim_isis = plot_data['sim_isis']
    real_band_powers = plot_data['real_band_powers']
    sim_band_powers = plot_data['sim_band_powers']
    sim_results = plot_data['sim_results']
    n_exc = sim_results['n_exc']

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 3, 1)
    plt.bar(['Real', 'Simulated'], [real_mean_rate, sim_mean_rate], 
            yerr=[real_std_rate, 0], capsize=5,
            color=['blue', 'green'], alpha=0.7)
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Firing Rate Comparison')

    plt.subplot(3, 3, 2)
    if len(real_isis) > 0:
        plt.hist(real_isis*1000, bins=60, alpha=0.6, label='Real', density=True, color='blue')
    if len(sim_isis) > 0:
        plt.hist(np.array(sim_isis)*1000, bins=60, alpha=0.6, label='Simulated', density=True, color='green')
    plt.xlabel('ISI (ms)')
    plt.ylabel('Density')
    plt.title('ISI Distribution')
    plt.legend()
    plt.xlim(0, 500)

    plt.subplot(3, 3, 3)
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    real_powers = [real_band_powers.get(band, 1e-10) for band in bands]
    sim_powers = [sim_band_powers.get(band, 1e-10) for band in bands]
    x = np.arange(len(bands))
    width = 0.35
    plt.bar(x - width/2, real_powers, width, label='Real', alpha=0.7, color='blue')
    plt.bar(x + width/2, sim_powers, width, label='Simulated', alpha=0.7, color='green')
    plt.xlabel('Frequency Band')
    plt.ylabel('Power')
    plt.title('LFP Band Power')
    plt.xticks(x, bands)
    plt.legend()
    plt.yscale('log')

    plt.subplot(3, 3, 4)
    spike_mon_exc = sim_results['spike_mon_exc']
    spike_mon_inh = sim_results['spike_mon_inh']
    time_mask = spike_mon_exc.t/ms < 2000
    plt.plot(spike_mon_exc.t[time_mask]/ms, spike_mon_exc.i[time_mask], '.g', markersize=0.8, alpha=0.7)
    time_mask_inh = spike_mon_inh.t/ms < 2000
    plt.plot(spike_mon_inh.t[time_mask_inh]/ms, spike_mon_inh.i[time_mask_inh] + n_exc, '.r', markersize=1.0)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Network Raster Plot')
    plt.xlim(0, 2000)

    plt.subplot(3, 3, 5)
    rate_mon_exc = sim_results['rate_mon_exc']
    rate_mon_inh = sim_results['rate_mon_inh']
    if len(rate_mon_exc.t) > 0:
        plt.plot(rate_mon_exc.t/ms, rate_mon_exc.smooth_rate(window='gaussian', width=50*ms)/Hz, color='green', linewidth=2, label='Excitatory')
    if len(rate_mon_inh.t) > 0:
        plt.plot(rate_mon_inh.t/ms, rate_mon_inh.smooth_rate(window='gaussian', width=50*ms)/Hz, color='red', linewidth=2, label='Inhibitory')
    plt.xlabel('Time (ms)')
    plt.ylabel('Population Rate (Hz)')
    plt.title('Population Rates')
    plt.legend()

    plt.subplot(3, 3, 6)
    state_mon_exc = sim_results['state_mon_exc']
    for i in range(min(4, state_mon_exc.v.shape[0])):
        plt.plot(state_mon_exc.t/ms, state_mon_exc.v[i]/mV, alpha=0.8, linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Membrane Voltages')

    plt.tight_layout()
