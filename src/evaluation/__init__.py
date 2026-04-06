"""Evaluation module for model assessment."""
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, roc_auc_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance and generate reports."""
    
    def __init__(self, model, X_test, y_test):
        """Initialize evaluator."""
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_proba = model.predict_proba(X_test)[:, 1]
    
    def generate_report(self, output_dir: str = 'reports/figures/'):
        """Generate comprehensive evaluation report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrix
        self._plot_confusion_matrix(output_path / 'confusion_matrix.png')
        
        # ROC Curve
        self._plot_roc_curve(output_path / 'roc_curve.png')
        
        # Precision-Recall Curve
        self._plot_pr_curve(output_path / 'pr_curve.png')
        
        # Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance(output_path / 'feature_importance.png')
        
        # Classification Report
        report = classification_report(self.y_test, self.y_pred)
        with open(output_path / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {output_dir}")
    
    def _plot_confusion_matrix(self, filepath):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
    
    def _plot_roc_curve(self, filepath):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        auc = roc_auc_score(self.y_test, self.y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
    
    def _plot_pr_curve(self, filepath):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
    
    def _plot_feature_importance(self, filepath, top_n: int = 15):
        """Plot feature importance."""
        importance = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance)), importance['importance'], align='center')
        plt.yticks(range(len(importance)), importance['feature'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()