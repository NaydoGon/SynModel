from brian2 import mV

def apply_dopamine_effect(synapse, dopamine_level, base_A_pre, base_A_post):
    if 'A_pre' not in synapse.namespace or 'A_post' not in synapse.namespace:
        raise ValueError("Synapse namespace must contain 'A_pre' and 'A_post' to apply dopamine effect.")
        
    synapse.namespace['A_pre'] = base_A_pre * (1 + dopamine_level)
    synapse.namespace['A_post'] = base_A_post * (1 + dopamine_level)
    print(f"Dopamine at level {dopamine_level}: A_pre={synapse.namespace['A_pre']}, A_post={synapse.namespace['A_post']}")

def apply_acetylcholine_effect(neuron_group, acetylcholine_level, base_v_rest):
    if 'v_rest' not in neuron_group.namespace:
        raise ValueError("Neuron group namespace must contain 'v_rest' to apply acetylcholine effect.")
        
    neuron_group.namespace['v_rest'] = base_v_rest + acetylcholine_level * 5 * mV
