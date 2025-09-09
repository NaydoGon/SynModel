from brian2 import mV, ms, nA, ohm, pA, volt, second, nS, pF

LIF_EQS = '''
dv/dt = (v_rest - v + I_syn) / tau : volt (unless refractory)
I_syn : volt
'''
LIF_PARAMS = {
    'v_rest': -70 * mV,
    'v_reset': -65 * mV,
    'v_thresh': -50 * mV,
    'tau': 10 * ms,
    'refractory_period': 5 * ms
}

EXC_EQS = '''
dv/dt = (v_rest - v + I_syn + I_noise - w) / tau_m_exc + sigma * xi * tau_m_exc**-0.5 : volt (unless refractory)
dw/dt = -w / tau_w : volt
I_syn : volt
I_noise : volt
sigma : volt
tau_w : second
'''

INH_EQS = '''
dv/dt = (v_rest - v + I_syn + I_noise) / tau_m_inh + sigma * xi * tau_m_inh**-0.5 : volt (unless refractory)
I_syn : volt
I_noise : volt
sigma : volt
'''

NETWORK_PARAMS = {
    'v_rest': -65 * mV,
    'v_reset': -70 * mV,
    'v_thresh': -50 * mV,
    'tau_m_exc': 25 * ms,
    'tau_m_inh': 15 * ms,
    'refractory_period': 3 * ms,
    'synaptic_weight': 0.6 * mV
}

ADEX_EQS = '''
dv/dt = (g_L * (E_L - v) + g_L * Delta_T * exp((v - v_thresh) / Delta_T) - w + I) / C : volt (unless refractory)
dw/dt = (a * (v - E_L) - w) / tau_w : amp
I : amp
'''
ADEX_PARAMS = {
    'g_L': 30 * nS,
    'E_L': -70.6 * mV,
    'v_thresh': -50.4 * mV,
    'v_reset': -70.6 * mV,
    'Delta_T': 2 * mV,
    'C': 281 * pF,
    'tau_w': 144 * ms,
    'a': 4 * nS,
    'b': 0.0805 * nA,
    'refractory_period': 4 * ms
}