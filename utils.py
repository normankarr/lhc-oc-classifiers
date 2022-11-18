import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.metrics import roc_curve, roc_auc_score


def load_data(input_frame):
    dataset = input_frame[["mj1","tau1j1","tau2j1","mj2","tau1j2","tau2j2"]]
    dataset["mjj"] = (((input_frame["pxj1"]**2+input_frame["pyj1"]**2+input_frame["pzj1"]**2+input_frame["mj1"]**2)**0.5+(input_frame["pxj2"]**2+input_frame["pyj2"]**2+input_frame["pzj2"]**2+input_frame["mj2"]**2)**0.5)**2-(input_frame["pxj1"]+input_frame["pxj2"])**2-(input_frame["pyj1"]+input_frame["pyj2"])**2-(input_frame["pzj1"]+input_frame["pzj2"])**2)**0.5/1000.
    dataset["mjTwo"] = dataset[["mj1", "mj2"]].max(axis=1)
    dataset["mjOne"] = dataset[["mj1", "mj2"]].min(axis=1)
    dataset["mjDelta"] = dataset["mjTwo"] - dataset["mjOne"]
    dataset["tau1jOne"] = (dataset["mjOne"] == dataset["mj1"])*dataset["tau1j1"]+(dataset["mjOne"] == dataset["mj2"])*dataset["tau1j2"]
    dataset["tau2jOne"] = (dataset["mjOne"] == dataset["mj1"])*dataset["tau2j1"]+(dataset["mjOne"] == dataset["mj2"])*dataset["tau2j2"]
    dataset["tau1jTwo"] = (dataset["mjTwo"] == dataset["mj1"])*dataset["tau1j1"]+(dataset["mjTwo"] == dataset["mj2"])*dataset["tau1j2"]
    dataset["tau2jTwo"] = (dataset["mjTwo"] == dataset["mj1"])*dataset["tau2j1"]+(dataset["mjTwo"] == dataset["mj2"])*dataset["tau2j2"]
    dataset["tau21jOne"] = dataset["tau2jOne"]/dataset["tau1jOne"]
    dataset["tau21jTwo"] = dataset["tau2jTwo"]/dataset["tau1jTwo"]
    dataset["mjTwo"] = dataset["mjTwo"]/1000.
    dataset["mjOne"] = dataset["mjOne"]/1000.
    dataset["mjDelta"] = dataset["mjDelta"]/1000.
    dataset = dataset.fillna(0)
    dataset = dataset[["mjj","mjOne","mjDelta","tau21jOne","tau21jTwo"]]
    
    return dataset.to_numpy()
    
def extract_regions(SR_low, SR_high, SB_low, SB_high, dataset_bg, dataset_sig):
    
    X_bg_SR = dataset_bg[(dataset_bg[:,0] > SR_low)*(dataset_bg[:,0] < SR_high)]
    X_sig_SR = dataset_sig[(dataset_sig[:,0] > SR_low)*(dataset_sig[:,0] < SR_high)]
    
    X_bg_SB = dataset_bg[(dataset_bg[:,0] > SB_low)*(dataset_bg[:,0] < SR_low)+(dataset_bg[:,0] > SR_high)*(dataset_bg[:,0] < SB_high)]
    X_sig_SB = dataset_sig[(dataset_sig[:,0] > SB_low)*(dataset_sig[:,0] < SR_low)+(dataset_sig[:,0] > SR_high)*(dataset_sig[:,0] < SB_high)]
    
    return (X_bg_SR, X_sig_SR), (X_bg_SB, X_sig_SB)

def plot_5_features(axes, x, label=None, histtype="bar", color=None):
    axes[0].set_xlabel("$m_{JJ}$ [GeV]")
    axes[0].set_yscale("log")
    axes[1].set_xlabel(r"$m_{J_{1}}$ [GeV]")
    axes[1].set_yscale("log")
    axes[2].set_xlabel(r"$m_{J_{2}}-m_{J_{1}}$ [GeV]")
    axes[2].set_yscale("log")
    axes[3].set_xlabel(r"$\tau_{21}^{J_{1}}$")
    axes[4].set_xlabel(r"$\tau_{21}^{J_{2}}$")
    
    axes[0].hist(x[:,0],bins=np.linspace(1,10,100),alpha=0.5,label=label, histtype=histtype,lw=3, color=color)
    #axes[0].legend(loc="upper right")
    for i in range(1,3):
        axes[i].hist(x[:,i],alpha=0.5,bins=np.linspace(-0.1,1,30), label=label, histtype=histtype,lw=3, color=color)
        #axes[i].legend(loc="upper right")
    for i in range(3,5):
        axes[i].hist(x[:,i],alpha=0.5,bins=np.linspace(-0.1,1,30), label=label, histtype=histtype,lw=3, color=color, density=True)
        #axes[i].legend(loc="upper right")
        
def split_data(data, train_frac=0.8):
    n = len(data)
    train_size = int(n*train_frac)
    train_data = data[:train_size,:]
    test_data = data[train_size:,:]
    return train_data, test_data


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