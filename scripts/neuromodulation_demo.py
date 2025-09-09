import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging
from src.logging_config import setup_logging
from tqdm import tqdm
import json

from brian2 import *
from brian2 import prefs

prefs.codegen.target = 'numpy'

from src.neuron_models import LIF_EQS, LIF_PARAMS
from src.synapses import STDP_EQS, STDP_PARAMS
from src.neuromodulation import apply_dopamine_effect, apply_acetylcholine_effect

def run_neuromodulation_demo(dopamine=0.0, acetylcholine=0.0):
    start_scope()
    
    model_ns = {
        'v_rest': LIF_PARAMS['v_rest'],
        'v_reset': LIF_PARAMS['v_reset'],
        'v_thresh': LIF_PARAMS['v_thresh'],
        'refractory_period': LIF_PARAMS['refractory_period'],
        'tau': LIF_PARAMS.get('tau', LIF_PARAMS.get('tau_m', 10*ms)),
        'tau_pre': STDP_PARAMS['tau_pre'],
        'tau_post': STDP_PARAMS['tau_post'],
        'A_pre': STDP_PARAMS['A_pre'],
        'A_post': STDP_PARAMS['A_post']
    }

    G = NeuronGroup(2, LIF_EQS,
                    threshold='v > v_thresh',
                    reset='v = v_reset',
                    refractory='refractory_period',
                    method='exact',
                    namespace=model_ns)
    
    v_rest = LIF_PARAMS['v_rest']
    v_thresh = LIF_PARAMS['v_thresh']
    G.v = v_rest + np.random.rand(2) * (v_thresh - v_rest)

    S = Synapses(G, G, STDP_EQS,
                 on_pre=STDP_PARAMS['on_pre'],
                 on_post=STDP_PARAMS['on_post'],
                 method='exact',
                 namespace=model_ns)
    S.connect(i=0, j=1)
    S.w = 0.5
    
    base_A_pre = STDP_PARAMS['A_pre']
    base_A_post = STDP_PARAMS['A_post']
    apply_dopamine_effect(S, dopamine_level=dopamine, base_A_pre=base_A_pre, base_A_post=base_A_post)
    
    base_v_rest = LIF_PARAMS['v_rest']
    apply_acetylcholine_effect(G, acetylcholine_level=acetylcholine, base_v_rest=base_v_rest)

    input_spikes = SpikeGeneratorGroup(1, [0], [50]*ms)
    input_syn = Synapses(input_spikes, G, on_pre='v_post += 25*mV')
    input_syn.connect(i=0, j=0)

    state_mon = StateMonitor(G, 'v', record=True)
    syn_mon = StateMonitor(S, 'w', record=True)

    run(200*ms)

    return state_mon, syn_mon

def plot_neuromodulation_results(dopamine_levels, ach_levels, results):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for i, dap_level in enumerate(dopamine_levels):
        w_mon = results['dopamine'][i][1]
        axes[0].plot(w_mon.t/ms, w_mon.w[0], label=f'Dopamine = {dap_level}')
    axes[0].set_ylabel('Synaptic Weight (w)')
    axes[0].set_title('Effect of Dopamine on STDP')
    axes[0].legend()
    
    for i, ach_level in enumerate(ach_levels):
        v_mon = results['acetylcholine'][i][0]
        axes[1].plot(v_mon.t/ms, v_mon.v[1]/mV, label=f'ACh = {ach_level}')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Postsynaptic Voltage (mV)')
    axes[1].set_title('Effect of Acetylcholine on Excitability')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/neuromodulation_demo.png")
    plt.show()


if __name__ == "__main__":
    setup_logging()
    
    dopamine_levels = [0.0, 0.5, 1.0]
    ach_levels = [0.0, 0.5, 1.0]
    results = {'dopamine': [], 'acetylcholine': []}
    output_results = {'dopamine': {}, 'acetylcholine': {}}

    for dap in tqdm(dopamine_levels, desc="Dopamine Simulations"):
        v_mon, w_mon = run_neuromodulation_demo(dopamine=dap)
        results['dopamine'].append((v_mon, w_mon))
        output_results['dopamine'][dap] = {
            "time": list(w_mon.t/ms),
            "synaptic_weight": list(w_mon.w[0])
        }

    for ach in tqdm(ach_levels, desc="Acetylcholine Simulations"):
        v_mon, w_mon = run_neuromodulation_demo(acetylcholine=ach)
        results['acetylcholine'].append((v_mon, w_mon))
        output_results['acetylcholine'][ach] = {
            "time": list(v_mon.t/ms),
            "postsynaptic_voltage": list(v_mon.v[1]/mV)
        }

    plot_neuromodulation_results(dopamine_levels, ach_levels, results)

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, "neuromodulation_demo.json"), "w") as f:
        json.dump(output_results, f, indent=4)
