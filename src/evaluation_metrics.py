import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import re
import pandas as pd
import os
from tqdm import tqdm

# Utility function to format the score given by the LLM
def _extract_leading_digits(text):
    match = re.match(r"^\d+", text)
    if match:
        return match.group(0)
    else:
        return ""

### AUROC
def _compute_AUROC(y, uq_scores):
    fpr, tpr, thresholds = metrics.roc_curve(y, uq_scores)
    auroc = metrics.auc(fpr, tpr)
    return auroc

def plot_AUROC(df, method):
    y = df['acc']
    scores = df[method]
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'img/entailment_C_ecc_AUROC')

### AUARC
def _compute_AUARC(y, uq_scores):
    sorted_indices = np.argsort(uq_scores)
    y_sorted = y[sorted_indices]

    rejection_fractions = []
    accuracies = []
    n = len(y)
    # Compute accuracy as we reject increasingly low confident examples
    for k in range(n + 1):
        if k == n:
            acc = np.nan
        else:
            acc = np.mean(y_sorted[k:])
        rejection_fraction = k / n
        rejection_fractions.append(rejection_fraction)
        accuracies.append(acc)
    rejection_fractions = rejection_fractions[:-1]
    accuracies = accuracies[:-1]

    aurac = metrics.auc(rejection_fractions, accuracies)
    return aurac

def plot_AUARC(df, uq_method):
    y = df['acc']
    scores = df[uq_method]
    sorted_indices = np.argsort(scores)
    y_sorted = y[sorted_indices]

    rejection_fractions = []
    accuracies = []
    n = len(y)
    # Compute accuracy as we reject increasingly low confident examples
    for k in range(n + 1):
        if k == n:
            acc = np.nan
        else:
            acc = np.mean(y_sorted[k:])
        rejection_fraction = k / n
        rejection_fractions.append(rejection_fraction)
        accuracies.append(acc)
    rejection_fractions = rejection_fractions[:-1]
    accuracies = accuracies[:-1]

    aurac = metrics.auc(rejection_fractions, accuracies)
    # Compute baseline accuracy (no rejections)
    baseline_accuracy = np.mean(y)

    plt.title('Rejection Accuracy Curve')
    plt.plot(rejection_fractions, accuracies, 'b', label='AURAC = %0.2f' % aurac)
    plt.hlines(baseline_accuracy, xmin=0, xmax=1, colors='r', linestyles='dotted', label='Baseline Accuracy = %0.2f' % baseline_accuracy)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Rejection Fraction')
    plt.ylabel('Accuracy on Remaining Questions')
    plt.savefig(f'img/entailment_C_ecc_AUARC')


def compute_all_metrics(methods=['jaccard', 'levenshtein', 'entail', 'contra', 'sbert']):
    filenames = sorted(os.listdir('data/generated_responses'))[:1]
    for filename in tqdm(filenames):
        df_with_assessment = pd.read_parquet(f'data/assessed_responses/{filename}')
        df_with_confidence = pd.read_parquet(f'data/responses_with_confidence/{filename}')

        # first, get the dataframes, format
        df_with_assessment['scores'] = df_with_assessment['scores'].apply(_extract_leading_digits).astype(float)
        y = df_with_assessment['scores'] >= 70

        results = []
        # should first compute random and oracle
        for method in methods:
            for uq_calc in ['_U_EigV', '_U_Deg', '_C_Deg', '_U_Ecc', '_C_Ecc']:
                uq_method = method + uq_calc
                uq_scores = df_with_confidence[uq_method]
                results.append({
                    'uq_method': uq_method,
                    'auroc': _compute_AUROC(y, uq_scores),
                    'auarc': _compute_AUARC(y, uq_scores)
                })
        # returns a table with all methods+random+oracle
        pd.DataFrame(results).to_csv('results/results.csv')
