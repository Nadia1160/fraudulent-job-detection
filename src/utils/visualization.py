"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_results(summary_df, all_results_df, best_feature, best_classifier, best_row, df_ruleset):
    """
    Generate all visualization plots

    Args:
        summary_df (pd.DataFrame): Summary of best models
        all_results_df (pd.DataFrame): All results
        best_feature (str): Best feature set name
        best_classifier (str): Best classifier name
        best_row (pd.Series): Best model results
        df_ruleset (pd.DataFrame): Ruleset features dataframe

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Bar chart of best F1-scores
    if not summary_df.empty:
        feature_names = summary_df['Feature Class'].values
        f1_scores = summary_df['Weighted Avg. F1-Score'].values

        bars = axes[0, 0].bar(range(len(feature_names)), f1_scores, color='skyblue')
        axes[0, 0].set_xticks(range(len(feature_names)))
        axes[0, 0].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].set_title('Best F1-Score by Feature Class')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Heatmap of classifier performance
    if not all_results_df.empty:
        heatmap_data = all_results_df.pivot_table(
            values='F1-Score',
            index='Feature Set',
            columns='Classifier',
            aggfunc='mean'
        )

        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                        ax=axes[0, 1], cbar_kws={'label': 'F1-Score'})
            axes[0, 1].set_title('Classifier Performance Heatmap')
            axes[0, 1].set_xlabel('Classifier')
            axes[0, 1].set_ylabel('Feature Set')

    # 3. Confusion matrix for best model
    if best_row is not None:
        cm = confusion_matrix(best_row['y_test'], best_row['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=['Real', 'Identity', 'Corporate', 'MLM'])
        disp.plot(ax=axes[1, 0], cmap='Blues', values_format='d')
        axes[1, 0].set_title(f'Best Model: {best_feature}\n{best_classifier} (F1={best_row["F1-Score"]:.3f})')

    # 4. Feature importance for ruleset
    if best_row is not None and hasattr(best_row['Model'], 'feature_importances_'):
        importances = best_row['Model'].feature_importances_
        feature_names = df_ruleset.drop('type', axis=1).columns

        indices = np.argsort(importances)[-15:]

        axes[1, 1].barh(range(len(indices)), importances[indices])
        axes[1, 1].set_yticks(range(len(indices)))
        axes[1, 1].set_yticklabels([feature_names[i] for i in indices])
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title('Top 15 Ruleset Feature Importances')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels (list): Label names
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    return plt.gcf()

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """
    Plot feature importance for tree-based models

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): Feature names
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Model does not have feature_importances_")

    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='x')
    return plt.gcf()