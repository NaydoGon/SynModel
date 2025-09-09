from brian2 import ms

STDP_EQS = '''
w : 1
dapre/dt = -apre / tau_pre : 1 (event-driven)
dapost/dt = -apost / tau_post : 1 (event-driven)
'''

STDP_PARAMS = {
    'tau_pre': 20 * ms,
    'tau_post': 20 * ms,
    'A_pre': 0.01,
    'A_post': -0.01 * 1.05,
    'on_pre': '''
        v_post += w * 10*mV
        apre += A_pre
        w = clip(w + apost, 0, 1)
    ''',
    'on_post': '''
        apost += A_post
        w = clip(w + apre, 0, 1)
    '''
}
