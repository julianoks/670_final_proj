from evaluate import evaluate
import itertools
from collections import OrderedDict
import numpy as np
import datetime
import argparse

param_names = ['useIOU', 'useLBP', 'useClassFeature', 'useKalmanState']

def get_overall_f1(metrics):
    return float(metrics.split('\n')[-1].split()[1][:-1])

def get_results(n_datasets = 1):
    all_param_combinations = [
        OrderedDict(zip(param_names, truths))
        for truths in itertools.product([True, False], repeat=len(param_names))
    ]
    f1_scores = [
        get_overall_f1(evaluate(tracker_weight_params = dict(param), n_datasets=n_datasets, train=True, gif=False))
        for param in all_param_combinations
    ]
    params_to_scores = dict(zip((tuple(d.values()) for d in all_param_combinations), f1_scores))

    # calculate shapley values
    attributions = np.zeros((len(param_names),))

    for perm in itertools.permutations(range(len(param_names))):
        present = [0] * len(param_names)
        val = params_to_scores[tuple(present)]
        for p in perm:
            present[p] = 1
            next_val = params_to_scores[tuple(present)]
            attributions[p] += next_val - val
            val = next_val

    attributions /= np.math.factorial(len(param_names))
    shapley_values = dict(zip(param_names, attributions))

    best_params = all_param_combinations[np.argmax(f1_scores)]
    best_params_score = np.max(f1_scores)

    return shapley_values, best_params, best_params_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_datasets', type=int, default=1)
    args = parser.parse_args()
    print(args)

    shapley_values, best_params, best_params_score = get_results(n_datasets = args.n_datasets)
    results = (f'Shapley Values: {shapley_values}\n'
        + f'The best parameters were {best_params}, with f1 score = {best_params_score}')
    print(results)
    with open(f'eval/results/{datetime.datetime.now()}_shapley_experiment.txt', 'w') as f:
        f.write(results)

