from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


def make_plots(ax1, ax2, scores_arr, y_test, model_name, baseline=True):
    
    for i in range(len(scores_arr)):
        fpr, tpr, _ = roc_curve(y_test, scores_arr[i])
        ax1.plot(fpr, tpr, label="Network: {0}".format(i))
        ax2.plot(tpr, np.true_divide(tpr, np.sqrt(fpr)), label="Network: {0}".format(i))
    
    ax1.set_title(model_name + " ROC Curve")
    if baseline:
        ax1.plot([0,1],[0,1],'--',label="baseline")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()

    ax2.set_title(model_name + " SIC Curve")
    ax2.set_xlabel("True Positive Rate")
    ax2.set_ylabel("TPR/Sqrt(FPR)")
    ax2.legend()