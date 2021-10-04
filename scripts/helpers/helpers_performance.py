#!/usr/bin/python3
# helpers_performance.py
# Defines helper functions for studying models fit to the validation sample
# Zach Warner
# 4 September 2020

##### SET UP #####

### load modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

##### HELPER FUNCTIONS #####

### plot test loss and test accuracy curves
def plot_diagnostics(name, history, variable_name, n_epoch):
    # get epochs
    epochs = list(range(1, n_epoch + 1))
    epochs_ticks = epochs
    if len(epochs) > 25:
        epochs_ticks = np.quantile(epochs, interpolation = 'nearest', q = [0, .25, .5, .75, 1])
        for i in range(1, 3):
            epochs_ticks[i] = int(np.around(epochs_ticks[i]/5, decimals = 0)*5)
    # open the figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
    # subplot 1: accuracy
    ax1.set_title('Accuracy')
    ax1.plot(epochs, history.history['accuracy'],'b--')
    ax1.plot(epochs, history.history['val_accuracy'], 'g-')
    ax1.set(xlabel = 'Epoch', ylabel = 'Accuracy')
    ax1.legend(['Train', 'Validation'], loc = 'lower right')
    ax1.set_xticks(epochs_ticks)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # subplot 2: loss
    ax2.set_title('Loss')
    ax2.plot(epochs, history.history['loss'], 'b--')
    ax2.plot(epochs, history.history['val_loss'], 'g-')
    ax2.set(xlabel = 'Epoch', ylabel = 'Loss')
    ax2.legend(['Train', 'Validation'], loc = 'upper right')
    ax2.set_xticks(epochs_ticks)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # subplot 3: MSE
    ax3.set_title('MSE')
    ax3.plot(epochs, history.history['mse'], 'b--')
    ax3.plot(epochs, history.history['val_mse'], 'g-')
    ax3.set(xlabel = 'Epoch', ylabel = 'MSE')
    ax3.legend(['Train', 'Validation'], loc = 'upper right')
    ax3.set_xticks(epochs_ticks)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # clean up and export
    fig.tight_layout(pad=.5)
    fig.set_figheight(9.69)
    fig.set_figwidth(6.27)
    plt.savefig(name)
    plt.close()

def crosstab(y_true, y_pred):
    # set up storage
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    # count each condition
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if y_true[i] == y_pred[i]:
                true_positives += 1
            else:
                false_positives += 1
        else: 
            if y_true[i] == y_pred[i]:
                true_negatives += 1
            else:
                false_negatives += 1
    return true_positives, true_negatives, false_positives, false_negatives

def classification_metrics(y_true, y_pred):
    # get the crosstabs
    true_positives, true_negatives, false_positives, false_negatives = crosstab(y_true, y_pred)
    # create measures
    recall = true_positives / (true_positives + false_negatives)
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 'NA'
    try:
        npv = true_negatives / (true_negatives + false_negatives)
    except ZeroDivisionError:
        npv = 'NA'
    try:
        specificity = true_negatives / (true_negatives + false_positives)
    except ZeroDivisionError:
        specificity = 'NA'
    accuracy = (true_positives + true_negatives) / (true_negatives + true_positives + false_negatives + false_positives)
    try:
        bal_accuracy = (recall + specificity)/2
    except TypeError:
        bal_accuracy = 'NA'
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except (TypeError, ZeroDivisionError):
        f1 = 'NA'
    # create a dictionary
    perf = {'recall': recall, 'precision': precision, 'neg_pred_value': npv, 'specificity': specificity, 'accuracy': accuracy, 'balanced_accuracy': bal_accuracy, 'f1': f1}
    # return the dictionary
    return perf
