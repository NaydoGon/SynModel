import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging_config import setup_logging
setup_logging()

import logging
import json

from brian2 import *
from brian2 import prefs

prefs.codegen.target = 'numpy'

from src.neuron_models import LIF_EQS, LIF_PARAMS
from src.stimuli import generate_oscillatory_input
from src.analysis import (analyze_lfp_bands, 
                          compute_coherence, infer_cognitive_state, bandpass_filter)

def run_analysis_simulation(duration=2*second):
    start_scope()

    n_neurons = 100
    duration = 2*second

    model_ns = {
        'v_rest': LIF_PARAMS['v_rest'],
        'v_reset': LIF_PARAMS['v_reset'],
        'v_thresh': LIF_PARAMS['v_thresh'],
        'refractory_period': LIF_PARAMS['refractory_period'],
        'tau': LIF_PARAMS.get('tau', LIF_PARAMS.get('tau_m', 10*ms))
    }

    layer4 = NeuronGroup(n_neurons, LIF_EQS,
                         threshold='v > v_thresh',
                         reset='v = v_reset',
                         refractory='refractory_period',
                         method='exact',
                         namespace=model_ns)
    
    v_rest = LIF_PARAMS['v_rest']
    v_thresh = LIF_PARAMS['v_thresh']
    layer4.v = v_rest + np.random.rand(n_neurons) * (v_thresh - v_rest)

    theta_drive = generate_oscillatory_input(6*Hz, duration)
    gamma_drive = generate_oscillatory_input(40*Hz, duration)
    input_group = PoissonGroup(n_neurons, rates='60*Hz + 25*Hz*theta_drive(t) + 15*Hz*gamma_drive(t)')
    
    input_syn = Synapses(input_group, layer4, on_pre='v_post += 1.8 * mV')
    input_syn.connect(p=0.2)
    
    spike_mon = SpikeMonitor(layer4)
    state_mon = StateMonitor(layer4, 'v', record=True)

    net = Network(collect())
    net.run(duration, report='text')
    
    return spike_mon, state_mon

def perform_and_plot_cognitive_analysis(spike_mon, state_mon):
    logging.info("Performing cognitive analysis and plotting results...")
    
    lfp = np.mean(state_mon.v / mV, axis=0)
    fs = float(1.0 / (defaultclock.dt / second))
    
    analysis_results = analyze_lfp_bands(lfp, fs)
    if analysis_results is None:
        logging.error("LFP analysis failed. Cannot proceed with cognitive analysis.")
        return None
    
    freqs, psd, band_powers = analysis_results
    theta_power = band_powers.get('theta', 0)
    gamma_power = band_powers.get('gamma', 0)

    total_spikes = len(spike_mon.t)
    num_neurons = len(spike_mon.source)
    duration_seconds = float(state_mon.t[-1] / second)
    mean_firing_rate = total_spikes / (num_neurons * duration_seconds) if (num_neurons * duration_seconds) > 0 else 0
    
    inferred_state = infer_cognitive_state(theta_power, gamma_power, mean_firing_rate)

    logging.info("\nCognitive Analysis Results:")
    logging.info(f"  - Mean Firing Rate: {mean_firing_rate:.2f} Hz")
    for band, power in band_powers.items():
        logging.info(f"  - {band.capitalize()} Power: {power:.4g}")
    logging.info(f"  - Inferred Cognitive State: {inferred_state}")

    try:
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
        
        axs[0].plot(freqs, psd)
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('PSD')
        axs[0].set_title('Power Spectral Density')
        axs[0].set_xlim(0, 100)
        
        axs[1].bar(band_powers.keys(), band_powers.values(), color='skyblue')
        axs[1].set_title('LFP Power in Frequency Bands')
        axs[1].set_ylabel('Power')
        axs[1].set_yscale('log')
        
        axs[2].plot(spike_mon.t/ms, spike_mon.i, '.k', markersize=0.5)
        axs[2].set_xlabel('Time (ms)')
        axs[2].set_ylabel('Neuron Index')
        axs[2].set_title(f'Spike Raster - Cognitive State: {inferred_state}')

        plt.suptitle('Cognitive Analysis Results')

        figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
            
        plt.savefig(os.path.join(figures_dir, "cognitive_analysis.png"))
        plt.show()
        
    except Exception as e:
        logging.error(f"Plotting failed: {e}")
        print("Analysis completed but plotting failed. Results printed above.")

    return {
        "mean_firing_rate": mean_firing_rate,
        "band_powers": band_powers,
        "inferred_state": inferred_state
    }

if __name__ == "__main__":
    spike_mon, state_mon = run_analysis_simulation()

    analysis_results = perform_and_plot_cognitive_analysis(spike_mon, state_mon)

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, "run_cognitive_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=4)