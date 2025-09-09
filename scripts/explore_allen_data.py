import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging
from src.logging_config import setup_logging
from src.allen_data import get_session_data
import json
import pandas as pd

def explore_allen_data():
    logging.info("Fetching Allen Brain Observatory session data...")
    try:
        session = get_session_data()
        logging.info(f"Successfully fetched session {session.ecephys_session_id}.")
    except Exception as e:
        logging.error(f"Failed to fetch session data: {e}")
        logging.error("Please check AllenSDK configuration and internet connection.")
        return

    if len(session.probes) == 0:
        logging.warning(f"No probes found for session {session.ecephys_session_id}. Skipping LFP analysis.")
        return

    probe_id = session.probes.index.values[0]
    logging.info(f"Using first available probe: {probe_id}.")

    try:
        lfp = session.get_lfp(probe_id)
        logging.info(f"LFP data successfully retrieved for probe {probe_id}.")
    except Exception as e:
        logging.error(f"Could not retrieve LFP data for probe {probe_id}: {e}")
        return
    
    try:
        logging.info(f"Session has {len(session.units)} units (neurons).")

        plt.figure(figsize=(8, 6))
        session.units['firing_rate'].hist(bins=50)
        plt.xlabel("Firing rate (Hz)")
        plt.ylabel("Neuron count")
        plt.title("Distribution of firing rates")
        plt.grid(False)
        plt.savefig("figures/firing_rate_distribution.png")
        plt.show()

        suitable_units = session.units.query("firing_rate > 1 and firing_rate < 10")
        if len(suitable_units) == 0:
            logging.warning("No units found with firing rate between 1-10 Hz. Using first available unit.")
            example_unit_id = session.units.index[0]
        else:
            example_unit_id = suitable_units.index[0]
            
        plt.figure(figsize=(10, 3))
        plt.eventplot(session.spike_times[example_unit_id], colors="black")
        plt.xlabel("Time (s)")
        plt.ylabel("Spikes")
        plt.title(f"Raster plot for unit {example_unit_id}")
        plt.savefig(f"figures/raster_plot_unit_{example_unit_id}.png")
        plt.show()

        try:
            lfp_time = lfp.coords["time"].values
            lfp_data = lfp.values[:, 0]
            
            fs = lfp.sampling_rate if hasattr(lfp, 'sampling_rate') else 1250
            end_sample = int(fs)
            end_sample = min(end_sample, len(lfp_time))
            
            plt.figure(figsize=(10, 4))
            plt.plot(lfp_time[:end_sample], lfp_data[:end_sample])
            plt.xlabel("Time (s)")
            plt.ylabel("LFP (µV)")
            plt.title(f"LFP from Probe {probe_id} - Channel 0 (First Second)")
            plt.savefig(f"figures/lfp_probe_{probe_id}.png")
            plt.show()
            
        except Exception as e:
            logging.error(f"Could not plot LFP data: {e}")

        try:
            try:
                stim_table = session.get_stimulus_table("natural_scenes")
            except Exception as e:
                logging.warning(f"Could not get natural_scenes stimulus table: {e}")
                available_stimuli = session.stimulus_names
                logging.info(f"Available stimulus types: {available_stimuli}.")
                if available_stimuli:
                    stim_type = available_stimuli[0]
                    logging.info(f"Using '{stim_type}' stimulus instead.")
                    stim_table = session.get_stimulus_table(stim_type)
                else:
                    stim_table = pd.DataFrame()
                    
            if not stim_table.empty:
                first_stim_onset = stim_table.start_time.iloc[0]
                
                window_spikes = [t for t in session.spike_times[example_unit_id] 
                                 if first_stim_onset <= t <= first_stim_onset + 1.0]
                
                logging.info("\n--- Stimulus Response Example ---")
                logging.info(f"Unit {example_unit_id} fired {len(window_spikes)} spikes in the 1s "
                             f"window after the first stimulus.")
            else:
                logging.warning("No stimulus table found for this session.")
                
        except Exception as e:
            logging.warning(f"Could not perform stimulus response analysis: {e}")
            
    except Exception as e:
        logging.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    setup_logging()
    explore_allen_data()
     
    session = get_session_data()
    if session:
        mean_firing_rate = session.units['firing_rate'].mean()
        std_firing_rate = session.units['firing_rate'].std()
        num_units = len(session.units)

        results = {
            "session_id": session.ecephys_session_id,
            "mean_firing_rate": mean_firing_rate,
            "std_firing_rate": std_firing_rate,
            "num_units": num_units
        }

        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        with open(os.path.join(results_dir, "explore_allen_data.json"), "w") as f:
            json.dump(results, f, indent=4)