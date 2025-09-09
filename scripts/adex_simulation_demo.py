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

from src.neuron_models import ADEX_EQS, ADEX_PARAMS

def run_adex_simulation(duration=1*second):
    start_scope()

    model_ns = {k: v for k, v in ADEX_PARAMS.items()}

    adex_group = NeuronGroup(1, ADEX_EQS,
                             threshold='v > v_thresh',
                             reset='v = v_reset; w += b',
                             refractory='refractory_period',
                             method='exponential_euler',
                             namespace=model_ns)
    
    v_rest = ADEX_PARAMS['E_L']
    v_thresh = ADEX_PARAMS['v_thresh']
    adex_group.v = v_rest 
    adex_group.w = 0 * nA

    input_current = 0.5 * nA
    adex_group.I = input_current
    
    state_mon = StateMonitor(adex_group, ['v', 'w'], record=True)
    spike_mon = SpikeMonitor(adex_group)

    net = Network(collect())
    net.run(200*ms, report='text')
    
    return spike_mon, state_mon

def plot_adex_results(spike_mon, state_mon):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True,
                            gridspec_kw={'height_ratios': [3, 1, 1]})

    axs[0].plot(state_mon.t/ms, state_mon.v[0]/mV, label='Voltage (v)')
    axs[0].set_ylabel('Voltage (mV)')
    axs[0].set_title('Membrane Potential and Adaptation Current')
    
    ax2 = axs[0].twinx()
    ax2.plot(state_mon.t/ms, state_mon.w[0]/nA, 'r--', label='Adaptation (w)')
    ax2.set_ylabel('Adaptation Current (nA)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

    axs[1].plot(state_mon.t/ms, np.ones_like(state_mon.t) * 0.5, 'g-')
    axs[1].set_ylabel('Input Current (nA)')
    axs[1].set_ylim(0, 1)

    axs[2].plot(spike_mon.t/ms, spike_mon.i, '.k')
    axs[2].set_xlabel('Time (ms)')
    axs[2].set_ylabel('Neuron')
    axs[2].set_yticks([])

    plt.suptitle('Adaptive Exponential (AdEx) Neuron Dynamics', fontsize=16)
    plt.savefig("figures/adex_simulation_demo.png")
    plt.show()

if __name__ == "__main__":
    setup_logging()
    spike_mon, state_mon = run_adex_simulation()

    plot_adex_results(spike_mon, state_mon)

    results = {
        "spike_times": list(spike_mon.t/ms),
        "neuron_indices": list(spike_mon.i),
        "voltage": list(state_mon.v[0]/mV),
        "adaptation_current": list(state_mon.w[0]/nA),
        "time": list(state_mon.t/ms)
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, "adex_simulation_demo.json"), "w") as f:
        json.dump(results, f, indent=4)
