import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm

# Utility function to format the score given by the LLM
def _extract_leading_digits(text):
    match = re.match(r"^\d+", text)
    if match:
        return match.group(0)
    else:
        print(f'Problem: {text}')
        return 0 # invalidate the response otherwise

def _plot_AUARC(rejection_fractions, accuracies, label):
    if 'oracle' in label:
        plt.plot(rejection_fractions, accuracies, 'b--', label=label)
    else:
        plt.plot(rejection_fractions, accuracies, label=label)

def _compute_AUARC(y, uq_scores, mode, label, plot=True): # mode is C or U
    if mode == 'C':
        sorted_indices = np.argsort(uq_scores)
    if mode == 'U':
        sorted_indices = np.argsort(-uq_scores)
    y_sorted = y[sorted_indices]

    rejection_fractions = []
    accuracies = []
    n = len(y)
    for k in range(n):
        acc = np.mean(y_sorted[k:]) if len(y_sorted[k:]) > 0 else 1.0
        rejection_fraction = k / n
        rejection_fractions.append(rejection_fraction)
        accuracies.append(acc)

    rejection_fractions.append(1.0)
    accuracies.append(1.0)

    if plot:
        _plot_AUARC(rejection_fractions, accuracies, label)

    aurac = metrics.auc(rejection_fractions, accuracies)
    return aurac

def compute_all_metrics(NUM_ANSWERS, plot=True, methods=['jaccard', 'levenshtein', 'entail', 'contra', 'sbert']):
    filenames = sorted(os.listdir('data/generated_responses'))

    scores = []
    for filename in tqdm(filenames):
        df_with_assessment = pd.read_parquet(f'data/assessed_responses/{filename}')
        df_with_assessment['scores'] = df_with_assessment['scores'].apply(_extract_leading_digits).astype(float)
        y = (df_with_assessment['scores'] >= 70).astype(int)
        scores.extend(y.to_list())

    y = np.array(scores)
    y_reshaped = y.reshape(-1, NUM_ANSWERS)
    expected_accuracies = y_reshaped.mean(axis=1)

    if plot:
        plt.figure(figsize=(6,4))
        plt.xlabel('Rejection Rate')
        plt.ylabel('Average Accuracy')
        plt.title('ARC')

    results = []
    for method in tqdm(methods):
        if method == 'entail':
            plot_method=True
        else:
            plot_method=False

        for uq_calc in ['_U_EigV', '_U_Deg', '_U_Ecc']: # uncertainty, expected accuracy
            uq_method = method + uq_calc

            uq_scores = []
            for filename in filenames:
                df_with_confidence = pd.read_parquet(f'data/responses_with_confidence/{filename}')                
                uq_scores.extend(df_with_confidence[uq_method].to_list())

            uq_scores = np.array(uq_scores)[::NUM_ANSWERS]
            results.append({
                'uq_method': uq_method,
                'auarc': _compute_AUARC(expected_accuracies, uq_scores, 'U', label=uq_method, plot=plot_method)
            })

        for uq_calc in ['_C_Deg', '_C_Ecc']: # confidence, individual accuracy
            uq_method = method + uq_calc

            uq_scores = []
            for filename in filenames:
                df_with_confidence = pd.read_parquet(f'data/responses_with_confidence/{filename}')                
                uq_scores.extend(df_with_confidence[uq_method].to_list())

            uq_scores = np.array(uq_scores)
            results.append({
                'uq_method': uq_method,
                'auarc': _compute_AUARC(y, uq_scores, 'C', label=uq_method, plot=plot_method)
            })

    results.append({
        'uq_method': 'random',
        'auarc': np.mean(y)
    })
    if plot:
        plt.plot(np.linspace(0, 1, 20), [np.mean(y)]*20, 'r--')

    results.append({
        'uq_method': 'oracle_U',
        'auarc': _compute_AUARC(expected_accuracies, expected_accuracies, 'C', label='oracle_U', plot=False)
    })

    results.append({
        'uq_method': 'oracle_C',
        'auarc': _compute_AUARC(y, y, 'C', label='oracle_C', plot=plot)
    })

    if plot:
        plt.legend()
        plt.savefig('img/AUARC.png')

    # returns a table with all methods+random+oracle
    pd.DataFrame(results).to_csv('results/results.csv', index=False)
