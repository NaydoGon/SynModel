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
from src.stimuli import generate_stimulus_sequence

def run_one_back_task_simulation(duration=1*second):
    start_scope()

    n_neurons = 20
    duration = 500*ms

    model_ns = {
        'v_rest': LIF_PARAMS['v_rest'],
        'v_reset': LIF_PARAMS['v_reset'],
        'v_thresh': LIF_PARAMS['v_thresh'],
        'refractory_period': LIF_PARAMS['refractory_period'],
        'tau': LIF_PARAMS.get('tau', LIF_PARAMS.get('tau_m', 10*ms))
    }

    layer4 = NeuronGroup(n_neurons, LIF_EQS,
                         threshold="v > v_thresh",
                         reset="v = v_reset",
                         refractory='refractory_period',
                         method='exact',
                         namespace=model_ns)
    
    v_rest = LIF_PARAMS['v_rest']
    v_thresh = LIF_PARAMS['v_thresh']
    layer4.v = v_rest + np.random.rand(n_neurons) * (v_thresh - v_rest)

    stimulus = generate_stimulus_sequence(duration)
    input_group = PoissonGroup(n_neurons, rates='200*Hz*stimulus(t)')
    syn = Synapses(input_group, layer4, on_pre='v_post += 1.5*mV')
    syn.connect(p=0.2)
    
    spike_mon = SpikeMonitor(layer4)
    state_mon = StateMonitor(layer4, 'v', record=True)
    
    net = Network(layer4, input_group, syn, spike_mon, state_mon)
    net.run(duration, report='text')
    
    return spike_mon, state_mon

def plot_one_back_task_results(spike_mon, state_mon):
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    plt.figure(figsize=(10, 4))
    plt.plot(spike_mon.t/ms, spike_mon.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title('Spike Raster for 1-Back Task')
    plt.savefig(os.path.join(figures_dir, "one_back_task_raster.png"))
    plt.show()
    
    lfp = np.mean(state_mon.v / mV, axis=0)
    plt.figure(figsize=(12, 4))
    plt.plot(state_mon.t/ms, lfp)
    plt.xlabel('Time (ms)')
    plt.ylabel('Mean Voltage (mV)')
    plt.title('Simulated LFP - Layer 4')
    plt.savefig(os.path.join(figures_dir, "one_back_task_lfp.png"))
    plt.show()

if __name__ == "__main__":
    setup_logging()
    spike_mon, state_mon = run_one_back_task_simulation()

    plot_one_back_task_results(spike_mon, state_mon)

    lfp = np.mean(state_mon.v / mV, axis=0)
    results = {
        "spike_times": [float(t) for t in spike_mon.t/ms],
        "neuron_indices": [int(i) for i in spike_mon.i],
        "lfp": [float(v) for v in lfp],
        "time": [float(t) for t in state_mon.t/ms]
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, "one_back_task_simulation.json"), "w") as f:
        json.dump(results, f, indent=4)
