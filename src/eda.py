import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc

from .utils import save_figure, print_separator


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return basic descriptive statistics for numeric columns.
    """
    print_separator("SUMMARY STATISTICS")
    stats = df.describe().T
    print(stats)
    return stats


def plot_histograms(
    df: pd.DataFrame,
    numeric_cols,
    filename: str = "histograms.png"
) -> None:
    """
    Plot histograms for selected numeric columns.
    """
    numeric_cols = list(numeric_cols)
    num_cols = len(numeric_cols)
    n_rows = (num_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=20)
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    save_figure(fig, filename)


def plot_boxplots(
    df: pd.DataFrame,
    numeric_cols,
    filename: str = "boxplots.png"
) -> None:
    """
    Plot boxplots for selected numeric columns.
    """
    numeric_cols = list(numeric_cols)
    num_cols = len(numeric_cols)
    n_rows = (num_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f"Boxplot of {col}")
        axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    save_figure(fig, filename)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    filename: str = "correlation_heatmap.png"
) -> None:
    """
    Plot a correlation heatmap for numeric features.
    """
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    save_figure(fig, filename)


def plot_confusion_matrix_heatmap(
    cm_df: pd.DataFrame,
    filename: str = "confusion_matrix.png"
) -> None:
    """
    Plot confusion matrix as a heatmap and save it.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    save_figure(fig, filename)


def plot_roc_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    filename: str = "roc_curve.png"
) -> None:
    """
    Plot ROC curve for a binary classifier and save it.
    """
    # probabilities for positive class (1 = Delayed)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    save_figure(fig, filename)
