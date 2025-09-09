import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging
from src.logging_config import setup_logging
import json
from brian2.units import *
import numpy

class Brian2Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Quantity, numpy.ndarray)):
            if isinstance(obj, Quantity) and obj.size == 1:
                return float(obj)
            return list(obj)
        if isinstance(obj, (numpy.int32, numpy.int64)):
            return int(obj)
        if isinstance(obj, (numpy.float32, numpy.float64)):
            return float(obj)
        return super(Brian2Encoder, self).default(obj)

from brian2 import *
from brian2 import prefs

prefs.codegen.target = 'numpy'

from src.allen_data import get_session_data, get_probe_data
from src.analysis import analyze_lfp_bands, analyze_isi_distribution, calculate_lfp
from src.neuron_models import EXC_EQS, INH_EQS, NETWORK_PARAMS
from src.plotting import plot_comparison

def run_simulation(real_data, duration=5*second):
    start_scope()
    
    n_exc = 120
    n_inh = 30

    model_ns = {
        'v_rest': NETWORK_PARAMS['v_rest'],
        'v_reset': NETWORK_PARAMS['v_reset'],
        'v_thresh': NETWORK_PARAMS['v_thresh'],
        'tau_m_exc': NETWORK_PARAMS['tau_m_exc'],
        'tau_m_inh': NETWORK_PARAMS['tau_m_inh'],
        'synaptic_weight': NETWORK_PARAMS['synaptic_weight'],
        'refractory_period': NETWORK_PARAMS['refractory_period']
    }

    excitatory = NeuronGroup(n_exc, EXC_EQS,
                            threshold='v > v_thresh',
                            reset='v = v_reset; w += 3*mV',
                            refractory='refractory_period',
                            method='euler',
                            namespace=model_ns)

    inhibitory = NeuronGroup(n_inh, INH_EQS,
                            threshold='v > v_thresh',
                            reset='v = v_reset',
                            refractory='refractory_period',
                            method='euler',
                            namespace=model_ns)

    excitatory.v = NETWORK_PARAMS['v_rest'] + (randn(n_exc) * 8 * mV)
    inhibitory.v = NETWORK_PARAMS['v_rest'] + (randn(n_inh) * 6 * mV)
    excitatory.sigma = 4.5 * mV
    inhibitory.sigma = 3.0 * mV
    excitatory.w = 0 * mV
    tau_w_values = 80 * ms + randn(n_exc) * 40 * ms
    excitatory.tau_w = np.clip(tau_w_values, 20*ms, 200*ms)

    ee_syn = Synapses(excitatory, excitatory, 'w_syn : volt', on_pre='I_syn_post += w_syn', namespace=model_ns)
    for i in range(n_exc):
        local_targets = list(range(max(0, i-10), min(n_exc, i+10)))
        if i in local_targets:
            local_targets.remove(i)
        n_local = int(len(local_targets) * 0.3)
        if n_local > 0:
            chosen_local = np.random.choice(local_targets, size=min(n_local, len(local_targets)), replace=False)
            for j in chosen_local:
                ee_syn.connect(i=i, j=j)
        
        distant_targets = [j for j in range(n_exc) if abs(i-j) > 10]
        n_distant = int(len(distant_targets) * 0.05)
        if n_distant > 0 and len(distant_targets) > 0:
            chosen_distant = np.random.choice(distant_targets, size=min(n_distant, len(distant_targets)), replace=False)
            for j in chosen_distant:
                ee_syn.connect(i=i, j=j)
                
    ee_syn.w_syn = NETWORK_PARAMS['synaptic_weight'] * (0.5 + 0.5 * rand(len(ee_syn)))

    ei_syn = Synapses(excitatory, inhibitory, 'w_syn : volt', on_pre='I_syn_post += w_syn', namespace=model_ns)
    ei_syn.connect(p=0.4)
    ei_syn.w_syn = NETWORK_PARAMS['synaptic_weight'] * 1.5 * (0.8 + 0.4 * rand(len(ei_syn)))

    ie_syn = Synapses(inhibitory, excitatory, 'w_syn : volt', on_pre='I_syn_post -= w_syn', delay=1*ms, namespace=model_ns)
    ie_syn.connect(p=0.6)
    ie_syn.w_syn = NETWORK_PARAMS['synaptic_weight'] * 3.0 * (0.7 + 0.6 * rand(len(ie_syn)))

    ii_syn = Synapses(inhibitory, inhibitory, 'w_syn : volt', on_pre='I_syn_post -= w_syn', namespace=model_ns)
    ii_syn.connect(p=0.2)
    ii_syn.w_syn = NETWORK_PARAMS['synaptic_weight'] * 2.0

    input_rate = real_data['mean_firing_rate'] * Hz
    input_neurons = PoissonGroup(n_exc, rates=input_rate)
    
    input_syn = Synapses(input_neurons, excitatory, on_pre='I_syn_post += synaptic_weight * 1.2', namespace=model_ns)
    
    n_connections = int(0.8 * n_exc)
    connect_indices = np.random.choice(n_exc, n_connections, replace=False)
    input_syn.connect(i=connect_indices, j=connect_indices)

    spike_mon_exc = SpikeMonitor(excitatory)
    spike_mon_inh = SpikeMonitor(inhibitory)
    state_mon_exc = StateMonitor(excitatory, ['v', 'I_syn', 'w'], record=range(min(30, n_exc)))
    rate_mon_exc = PopulationRateMonitor(excitatory)
    rate_mon_inh = PopulationRateMonitor(inhibitory)

    net = Network(collect())
    net.run(duration, report='text')

    return {
        "spike_mon_exc": spike_mon_exc,
        "spike_mon_inh": spike_mon_inh,
        "state_mon_exc": state_mon_exc,
        "rate_mon_exc": rate_mon_exc,
        "rate_mon_inh": rate_mon_inh,
        "duration": duration,
        "n_exc": n_exc,
    }

def main(args):
    setup_logging()
    
    session = get_session_data()
    real_data = get_probe_data(session)

    sim_results = run_simulation(real_data, duration=args.duration * second)
    
    real_isis = analyze_isi_distribution(real_data['spike_times'])
    _, _, real_band_powers = analyze_lfp_bands(real_data['lfp'], real_data['lfp_fs'])
    
    sim_spike_times = {i: sim_results['spike_mon_exc'].t[sim_results['spike_mon_exc'].i == i] for i in range(sim_results['n_exc'])}
    sim_isis = analyze_isi_distribution(sim_spike_times)
    
    sim_lfp = calculate_lfp(sim_results['state_mon_exc'])
    _, _, sim_band_powers = analyze_lfp_bands(sim_lfp, 1000.0)

    real_mean_rate = real_data['mean_firing_rate']
    real_std_rate = real_data['std_firing_rate']
    sim_mean_rate = len(sim_results['spike_mon_exc'].t) / (sim_results['n_exc'] * sim_results['duration'])
    
    plot_data = {
        'real_mean_rate': real_mean_rate,
        'real_std_rate': real_std_rate,
        'sim_mean_rate': sim_mean_rate,
        'real_isis': real_isis,
        'sim_isis': sim_isis,
        'real_band_powers': real_band_powers,
        'sim_band_powers': sim_band_powers,
    }
    
    plot_comparison({**plot_data, 'sim_results': sim_results})

    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, "simulation_comparison.png"))
    plt.show()

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, "run_simulation.json"), "w") as f:
        json.dump(plot_data, f, indent=4, cls=Brian2Encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural simulation and compare with Allen data.")
    parser.add_argument('--duration', type=float, default=5.0, help='Duration of the simulation in seconds.')
    args = parser.parse_args()
    main(args)