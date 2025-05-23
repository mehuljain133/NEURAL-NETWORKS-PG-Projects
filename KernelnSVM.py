# Unit-III Kernel methods and support vector machines: soft margin techniques.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# --- Generate Data ---
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.5)
y = np.where(y == 0, -1, 1)  # Use -1, +1 for SVM compatibility

# --- 1. Linear SVM (Hard Margin) ---
clf_hard = svm.SVC(kernel='linear', C=1e5)  # Very large C => hard margin
clf_hard.fit(X, y)

# --- 2. Soft Margin SVM (Linear) ---
clf_soft = svm.SVC(kernel='linear', C=1.0)  # Lower C => soft margin
clf_soft.fit(X, y)

# --- 3. RBF Kernel SVM ---
clf_rbf = svm.SVC(kernel='rbf', gamma='scale', C=1.0)
clf_rbf.fit(X, y)

# --- Function to plot decision boundaries ---
def plot_decision_boundary(clf, X, y, title):
    plt.figure(figsize=(6,5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = clf.decision_function(xy).reshape(xx.shape)

    # Decision boundary and margins
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], linestyles=['--', '-', '--'])
    ax.set_title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Plot Results ---
plot_decision_boundary(clf_hard, X, y, "Hard Margin SVM (C=1e5)")
plot_decision_boundary(clf_soft, X, y, "Soft Margin SVM (C=1.0)")
plot_decision_boundary(clf_rbf, X, y, "SVM with RBF Kernel")

# --- Print Support Vectors Count ---
print("Support Vectors (Hard Margin):", len(clf_hard.support_))
print("Support Vectors (Soft Margin):", len(clf_soft.support_))
print("Support Vectors (RBF Kernel):", len(clf_rbf.support_))
