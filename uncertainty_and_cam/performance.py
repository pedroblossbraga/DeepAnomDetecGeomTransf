import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def save_roc_pr_curve_data(scores, labels, file_path):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision_norm': precision_norm,
        'recall_norm': recall_norm,
        'pr_auc_norm': pr_auc_norm,
        'precision_anom': precision_anom,
        'recall_anom': recall_anom,
        'pr_auc_anom': pr_auc_anom,

    }

def plot_roc_auc_comparison(
        perf_dirichlet,
        perf_entropy,
        perf_entropy_uncert,
        OUTPUT_DIR
    ):
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # perf = np.load(dirichlet_perf_path)
    # perf_entropy = np.load(entropy_perf_path)

    plt.figure(figsize=(6, 3))
    plt.title('Receiver Operating Characteristic')

    plt.plot(perf_dirichlet['fpr'], perf_dirichlet['tpr'], 'royalblue',
            #  linestyle='dashed',
            linewidth = 5,
            label = '(Dirichlet score) AUC = %0.2f' % perf_dirichlet['roc_auc'])
    
    plt.plot(perf_entropy['fpr'], perf_entropy['tpr'], 'darkorange',
            linestyle='dotted', linewidth = 5,
            label = '(Entropy score) AUC = %0.2f' % perf_entropy['roc_auc'])
    
    plt.plot(perf_entropy_uncert['fpr'], perf_entropy_uncert['tpr'], 'forestgreen',
            linestyle='dashed', linewidth = 5,
            label = '(Entropy-Uncertainty) AUC = %0.2f' % perf_entropy_uncert['roc_auc'])

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(OUTPUT_DIR, f'roc_auc_comparison.png'))
    plt.show()