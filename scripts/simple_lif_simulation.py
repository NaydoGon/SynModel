import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging
from src.logging_config import setup_logging
import json

from brian2 import *
from brian2 import prefs

prefs.codegen.target = 'numpy'

from src.neuron_models import LIF_EQS, LIF_PARAMS
from src.synapses import STDP_EQS, STDP_PARAMS
from src.stimuli import generate_oscillatory_input
from scipy.signal import welch

def run_simple_lif_simulation(duration=1*second):
    start_scope()
    
    n_neurons = 100
    
    objects = []
    
    model_ns = {
        'v_rest': LIF_PARAMS['v_rest'],
        'v_reset': LIF_PARAMS['v_reset'],
        'tau': LIF_PARAMS.get('tau', LIF_PARAMS.get('tau_m', 10*ms)),
        'v_thresh': LIF_PARAMS['v_thresh'],
        'refractory_period': LIF_PARAMS['refractory_period'],
        'tau_pre': STDP_PARAMS['tau_pre'],
        'tau_post': STDP_PARAMS['tau_post'],
        'A_pre': STDP_PARAMS['A_pre'],
        'A_post': STDP_PARAMS['A_post']
    }
    
    layers = {}
    for layer_name in ['L2_3', 'L4', 'L5', 'L6']:
        group = NeuronGroup(n_neurons, LIF_EQS, 
                            threshold="v > v_thresh", 
                            reset="v = v_reset",
                            refractory='refractory_period', 
                            method='exact', 
                            name=f'{layer_name}_neurons',
                            namespace=model_ns)
        v_rest = LIF_PARAMS['v_rest']
        v_thresh = LIF_PARAMS['v_thresh']
        group.v = v_rest + np.random.rand(n_neurons) * (v_thresh - v_rest)
        layers[layer_name] = group
        objects.append(group)

    for pre_name, pre_group in layers.items():
        for post_name, post_group in layers.items():
            syn = Synapses(pre_group, post_group,
                           model=STDP_EQS,
                           on_pre=STDP_PARAMS['on_pre'],
                           on_post=STDP_PARAMS['on_post'],
                           namespace=model_ns,
                           name=f'syn_{pre_name}_{post_name}')
            syn.connect(p=0.1)
            syn.w = 'rand()'
            objects.append(syn)
    
    theta_drive = generate_oscillatory_input(6*Hz, duration)
    gamma_drive = generate_oscillatory_input(40*Hz, duration)

    input_group = PoissonGroup(n_neurons, rates='50*Hz + 20*Hz*theta_drive(t) + 10*Hz*gamma_drive(t)')
    input_syn = Synapses(input_group, layers['L4'], on_pre='v_post += 1.5 * mV')
    input_syn.connect(p=0.2)
    objects.extend([input_group, input_syn])

    spike_mon = SpikeMonitor(layers['L4'])
    state_mon = StateMonitor(layers['L4'], 'v', record=True)
    objects.extend([spike_mon, state_mon])

    net = Network(objects)
    net.run(duration, report='text')

    return spike_mon, state_mon

def plot_simple_lif_results(spike_mon, state_mon):
    plt.figure(figsize=(10, 4))
    plt.plot(spike_mon.t/ms, spike_mon.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title('Spike Raster - Layer 4')
    plt.savefig("figures/simple_lif_simulation_spike_raster.png")
    plt.show()

    lfp = np.mean(state_mon.v / mV, axis=0)
    freqs, psd = welch(lfp, fs=1000.0)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(state_mon.t/ms, lfp)
    plt.xlabel('Time (ms)')
    plt.ylabel('Mean Voltage (mV)')
    plt.title('Simulated LFP - Layer 4')

    plt.subplot(122)
    plt.semilogy(freqs, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power Spectral Density')
    plt.tight_layout()
    plt.savefig("figures/simple_lif_simulation_lfp_analysis.png")
    plt.show()

if __name__ == "__main__":
    setup_logging()
    spike_mon, state_mon = run_simple_lif_simulation()
    
    plot_simple_lif_results(spike_mon, state_mon)

    lfp = np.mean(state_mon.v / mV, axis=0)
    freqs, psd = welch(lfp, fs=1000.0)

    results = {
        "spike_times": [float(t) for t in spike_mon.t/ms],
        "neuron_indices": [int(i) for i in spike_mon.i],
        "lfp": [float(v) for v in lfp],
        "lfp_time": [float(t) for t in state_mon.t/ms],
        "psd_frequencies": [float(f) for f in freqs],
        "psd_power": [float(p) for p in psd]
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    with open(os.path.join(results_dir, "simple_lif_simulation.json"), "w") as f:
        json.dump(results, f, indent=4)
