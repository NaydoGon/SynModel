import os
import json
import glob
import numpy as np

def analyze_and_combine_results(): 
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    all_results = {}
     
    print("Running main simulation to gather detailed statistics...")
    from run_simulation import run_simulation as run_main_sim
    from src.allen_data import get_session_data, get_probe_data
    from src.analysis import analyze_isi_distribution, compare_isi_distributions
    from brian2 import second
    
    session = get_session_data()
    real_data = get_probe_data(session)
    duration = 5 * second
    sim_results = run_main_sim(real_data, duration=duration)
 
    sim_spike_times = {i: sim_results['spike_mon_exc'].t[sim_results['spike_mon_exc'].i == i] 
                       for i in range(sim_results['n_exc'])}
    
    rates = [(len(times) / (duration / second)) for times in sim_spike_times.values()]
    sim_mean_rate_per_neuron = np.mean(rates)
    sim_std_rate_per_neuron = np.std(rates)
     
    sim_isis = analyze_isi_distribution(sim_spike_times)
    sim_cv_isi = np.std(sim_isis) / np.mean(sim_isis) if np.mean(sim_isis) > 0 else 0
    
    real_isis = analyze_isi_distribution(real_data['spike_times'])
    real_cv_isi = np.std(real_isis) / np.mean(real_isis) if np.mean(real_isis) > 0 else 0
 
    ks_stat, p_value = compare_isi_distributions(real_isis, sim_isis)

    main_sim_summary = {
        'sim_mean_rate': sim_mean_rate_per_neuron,
        'sim_std_rate': sim_std_rate_per_neuron,
        'sim_cv_isi': sim_cv_isi,
        'real_cv_isi': real_cv_isi,
        'real_mean_rate': real_data['mean_firing_rate'],
        'real_std_rate': real_data['std_firing_rate'],
        'ks_statistic': ks_stat,
        'p_value': p_value
    }
    all_results['run_simulation_summary'] = main_sim_summary
    print("Main simulation analysis complete.")
     
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        key_name = os.path.splitext(filename)[0]
         
        if key_name == 'run_simulation':
            continue
            
        try:
            with open(file_path, 'r') as f:
                all_results[key_name] = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filename}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not read {filename} due to {e}. Skipping.")

    return all_results

if __name__ == "__main__":
    combined_results = analyze_and_combine_results()
     
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'combined_results.json')
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=4)
        
    print(f"\nCombined results saved to {output_path}")
     
    summary = combined_results.get('run_simulation_summary', {})
    sim_mean = summary.get('sim_mean_rate', 'XX.X')
    sim_std = summary.get('sim_std_rate', 'Y.Y')
    ks_stat = summary.get('ks_statistic', 'N/A')
    p_value = summary.get('p_value', 'N/A')
     
    print(f"Simulated Mean Firing Rate: {sim_mean:.2f} \u00b1 {sim_std:.2f} Hz")
    print(f"ISI Distribution KS-test: statistic={ks_stat:.4f}, p-value={p_value:.4f}") 
