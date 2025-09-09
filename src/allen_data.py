import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import warnings
import logging
from tqdm import tqdm

def get_session_data(cache_dir="ecephys_cache", session_id=None):
    logging.info("Initializing AllenSDK cache...")
    manifest_path = os.path.join(cache_dir, "manifest.json")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    
    if session_id is None:
        logging.info("No session ID provided. Searching for available sessions...")
        sessions = cache.get_session_table()
        if sessions.empty:
            logging.error("No sessions found.")
            raise ValueError("No sessions found in the cache.")
        session_id = sessions.index[0]
        logging.info(f"Using session {session_id}.")
        
    logging.info(f"Fetching data for session {session_id}...")
    
    class AllenSDKProgressBar(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_update_value = 0

        def update_to(self, current_value, total_value):
            increment = current_value - self._last_update_value
            self.update(increment)
            self.total = total_value
            self._last_update_value = current_value

    with AllenSDKProgressBar(unit='B', unit_scale=True, miniters=1, 
                             desc=f"Downloading Session {session_id}") as pbar:
        session = cache.get_session_data(session_id)
        pbar.update_to(1, 1)

    logging.info("Session data downloaded.")
    return session

def get_probe_data(session, probe_id=None):
    if probe_id is None:
        probes = session.probes
        if probes.empty:
            logging.error("No probes found.")
            raise ValueError("No probes found in the session.")
        probe_id = probes.index[0]
        logging.info(f"No probe ID given. Using probe {probe_id}.")
        
    units = session.units
    probe_units = units[units['probe_id'] == probe_id]
    logging.info(f"Processing {len(probe_units)} units from probe {probe_id}...")
    
    firing_rates = []
    spike_times_dict = {}
    
    for unit_id in tqdm(probe_units.index, desc="Extracting spike times"):
        spikes = session.spike_times[unit_id]
        if len(spikes) > 1:
            recording_duration = spikes[-1] - spikes[0]
            firing_rate = len(spikes) / recording_duration
            firing_rates.append(firing_rate)
            spike_times_dict[unit_id] = spikes
            
    mean_firing_rate = np.mean(firing_rates) if firing_rates else 0.0
    std_firing_rate = np.std(firing_rates) if firing_rates else 0.0
    
    probe_data = {
        "mean_firing_rate": mean_firing_rate,
        "std_firing_rate": std_firing_rate,
        "spike_times": spike_times_dict,
    }

    logging.info(f"Retrieving LFP data for probe {probe_id}...")
    try:
        lfp = session.get_lfp(probe_id)
        logging.info("LFP data retrieved.")
    except Exception as e:
        logging.error(f"Failed to get LFP data for probe {probe_id}.")
        raise ConnectionError(f"Could not fetch LFP data from AllenSDK. Original error: {e}")

    probe_data['lfp'] = lfp.values.flatten()
    
    if hasattr(lfp, 'time') and len(lfp['time']) > 1:
        lfp_time = lfp['time'].values
        probe_data['lfp_fs'] = 1.0 / (lfp_time[1] - lfp_time[0])
    else:
        probe_data['lfp_fs'] = 1250.0 
    
    return probe_data
