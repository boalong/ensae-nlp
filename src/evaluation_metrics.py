import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

### AUROC

y = data['acc']
scores = data['entail_C_Ecc']
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

y = data['acc']
scores = data['entail_C_Ecc']
# Sort by confidence (lowest first)
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
