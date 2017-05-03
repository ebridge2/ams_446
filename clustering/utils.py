import numpy as np
import matplotlib.pyplot as plt


def plot_confusion(conf, norm_conf, true_classes,
                   pred_classes, cmap=plt.cm.jet):
    """
    A function to plot the confusion matrix for a given dataset.
    The number of points given the [i, j] label is the value in each
    box, and the normalized confusion score is shown by the color
    of each box.

    **Positional Arguments:**
        - conf:
            - the confusion matrix.
        - norm_conf:
            - the normalized confusion matrix.
        - true_classes:
            - the true labels.
        - pred_classes:
            - the predicted labels.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = ax.imshow(norm_conf, cmap=cmap, interpolation='nearest',
                    aspect='auto')
    ntrue, npred = conf.shape
    for i in xrange(ntrue):
        for j in xrange(npred):
            ax.annotate(str(conf[i, j]), xy=(j, i),
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.yticks(range(ntrue), true_classes)
    plt.xticks(range(npred), pred_classes)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    cb = fig.colorbar(res)
    fig.tight_layout()
    return fig

def confusion_matrix(M, D):
    """
    A utility to compute the confusion matrix for a clustering algorithm.
    Returns the raw confusion numbers, and also a normalized matrix
    where the value is the confusion normalized by the total number
    of points for a given cluster.

    **Positional Arguments:**
        - M:
            - the true labels.
        - D:
            - the predicted labels.
    """
    M = np.array(M)
    D = np.array(D)
    true_classes = np.unique(M)
    pred_classes = np.unique(D)
    # cmtx shows TP, FP, TN, FN
    cmtx = np.zeros((len(true_classes), len(pred_classes)))
    # pmtx shows the confusion matrix normalized
    pmtx = np.zeros((len(true_classes), len(pred_classes)))
    for i, pred in enumerate(pred_classes):
        # get the indices that should be labelled
        # true
        Mcol = M[D == pred]
        n = len(Mcol)
        for j, true in enumerate(true_classes):
            cmtx[i, j] = np.sum(Mcol == true)
            pmtx[i, j] = cmtx[i, j]/float(n)
    fig_conf = plot_confusion(cmtx, pmtx, M, D)
    return (cmtx, pmtx, fig_conf)

def purity(M, D):
    """
    A utility to compute the purity of a clustering algorithm.

    **Positional Arguments:**
        - M:
            - the true labels.
        - D:
            - the predicted labels.
    """
    (cmtx, pmtx, fig_conf) = confusion_matrix(M, D)
    purity = np.sum(cmtx.max(axis=1))/float(np.sum(cmtx))
    return (purity, cmtx, pmtx, fig_conf)

def plot_laplacian(L, cmap=plt.cm.jet):
    """
    A function to plot the graph laplacian for use with spectral
    clustering visualizations.

    **Positional Arguments:**
        - L:
            - the graph laplacian.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    minval = np.percentile(L, 0)
    # since the diagonal will be very significant, this
    # will eliminate it from our plots
    maxval = np.percentile(L, 95)
    res = ax.imshow(L, cmap=cmap, interpolation='nearest', vmin=minval,
                    vmax=maxval)
    ax.set_ylabel('Training Example')
    ax.set_xlabel('Training Example')
    ax.set_title('Graph Laplacian')
    cb = fig.colorbar(res)
    return fig
