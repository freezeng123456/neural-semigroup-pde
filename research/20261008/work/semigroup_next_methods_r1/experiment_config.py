"""Prospective arms; scheduled final endpoints are fixed before optimization."""
SPECS = {
    'first20e500': {'family': 'map', 'objective': 'first', 'width': 20, 'epochs': 500, 'calls': 6},
    'teacher20e500': {'family': 'map', 'objective': 'teacher2', 'width': 20, 'epochs': 500, 'calls': 7},
    'rollout20e500': {'family': 'map', 'objective': 'rollout2', 'width': 20, 'epochs': 500, 'calls': 7},
    'pushforward20e500': {'family': 'map', 'objective': 'pushforward2', 'width': 20, 'epochs': 500, 'calls': 7},
    'teacher40e500': {'family': 'map', 'objective': 'teacher2', 'width': 40, 'epochs': 500, 'calls': 7},
    'teacher20e1000': {'family': 'map', 'objective': 'teacher2', 'width': 20, 'epochs': 1000, 'calls': 7},
    'auto20e500': {'family': 'flow', 'autonomous': True, 'width': 20, 'epochs': 500, 'calls': 1},
    'time20e500': {'family': 'flow', 'autonomous': False, 'width': 20, 'epochs': 500, 'calls': 1},
}
ARMS = tuple(SPECS)
MAP_ARMS = tuple(k for k, v in SPECS.items() if v['family'] == 'map')
FLOW_ARMS = tuple(k for k, v in SPECS.items() if v['family'] == 'flow')


def matrix():
    tasks = []
    # Rotate arm order by seed, so each arm occupies different array positions.
    for seed in range(3):
        order = ARMS[seed:] + ARMS[:seed]
        for arm in order:
            spec = SPECS[arm]
            tasks.append({'index': len(tasks), 'arm': arm, 'seed': seed,
                          'run_name': f'{arm}-seed{seed}-r1', **spec,
                          'updates': spec['epochs']*100,
                          'total_forward_calls': spec['epochs']*100*spec['calls']})
    return {'status': 'PROSPECTIVE_FIXED_MATRIX', 'tasks': tasks,
            'map_tasks': 18, 'flow_tasks': 6, 'max_concurrency': 4,
            'gpu_per_task': 1, 'cpu_per_task': 10,
            'wall_seconds': 21600, 'selection': 'scheduled final only'}
