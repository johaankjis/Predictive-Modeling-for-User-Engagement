"""
Model Evaluation Module
Evaluates trained models and generates performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)
import pickle
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def load_models():
    """Load trained models"""
    print("[v0] Loading trained models...")
    
    model_names = ['logistic_regression', 'random_forest', 'gradient_boosting']
    models = {}
    
    for name in model_names:
        with open(f'models/{name}.pkl', 'rb') as f:
            models[name] = pickle.load(f)
        print(f"[v0] Loaded {name}")
    
    return models

def load_test_data():
    """Load test data"""
    print("[v0] Loading test data...")
    
    with open('models/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    X_test = np.array(test_data['X_test'])
    y_test = np.array(test_data['y_test'])
    
    print(f"[v0] Loaded test data: {len(y_test)} samples")
    return X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics
    """
    print(f"[v0] Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    print(f"[v0] {model_name} - Accuracy: {metrics['accuracy']:.4f}, "
          f"F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics, y_pred, y_pred_proba

def plot_confusion_matrices(all_metrics):
    """
    Plot confusion matrices for all models
    """
    print("[v0] Generating confusion matrix plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, metrics in enumerate(all_metrics):
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f"{metrics['model_name']}\nAccuracy: {metrics['accuracy']:.3f}")
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('public/plots/confusion_matrices.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("[v0] Confusion matrices saved")

def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for all models
    """
    print("[v0] Generating ROC curves...")
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.savefig('public/plots/roc_curves.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("[v0] ROC curves saved")

def plot_feature_importance(models, feature_names):
    """
    Plot feature importance for tree-based models
    """
    print("[v0] Generating feature importance plots...")
    
    # Load feature names
    with open('models/feature_names.json', 'r') as f:
        features = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest
    rf_importance = models['random_forest'].feature_importances_
    rf_indices = np.argsort(rf_importance)[-15:]  # Top 15 features
    
    axes[0].barh(range(len(rf_indices)), rf_importance[rf_indices], color='steelblue')
    axes[0].set_yticks(range(len(rf_indices)))
    axes[0].set_yticklabels([features[i] for i in rf_indices], fontsize=9)
    axes[0].set_xlabel('Importance', fontsize=11)
    axes[0].set_title('Random Forest - Top 15 Features', fontsize=12, fontweight='bold')
    
    # Gradient Boosting
    gb_importance = models['gradient_boosting'].feature_importances_
    gb_indices = np.argsort(gb_importance)[-15:]
    
    axes[1].barh(range(len(gb_indices)), gb_importance[gb_indices], color='coral')
    axes[1].set_yticks(range(len(gb_indices)))
    axes[1].set_yticklabels([features[i] for i in gb_indices], fontsize=9)
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].set_title('Gradient Boosting - Top 15 Features', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('public/plots/feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("[v0] Feature importance plots saved")
    
    # Save feature importance data
    importance_data = {
        'random_forest': {
            'features': [features[i] for i in rf_indices],
            'importance': [float(rf_importance[i]) for i in rf_indices]
        },
        'gradient_boosting': {
            'features': [features[i] for i in gb_indices],
            'importance': [float(gb_importance[i]) for i in gb_indices]
        }
    }
    
    with open('public/feature_importance.json', 'w') as f:
        json.dump(importance_data, f, indent=2)

def plot_metrics_comparison(all_metrics):
    """
    Plot comparison of all metrics across models
    """
    print("[v0] Generating metrics comparison plot...")
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = [m['model_name'] for m in all_metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for idx, metrics in enumerate(all_metrics):
        values = [metrics[m] for m in metrics_names]
        ax.bar(x + idx * width, values, width, label=metrics['model_name'])
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add target line at 0.85
    ax.axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target (85%)')
    
    plt.tight_layout()
    plt.savefig('public/plots/metrics_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("[v0] Metrics comparison saved")

def generate_evaluation_report(all_metrics):
    """
    Generate comprehensive evaluation report
    """
    print("[v0] Generating evaluation report...")
    
    # Find best model
    best_model = max(all_metrics, key=lambda x: x['accuracy'])
    
    report = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'models_evaluated': len(all_metrics),
        'best_model': best_model['model_name'],
        'best_accuracy': best_model['accuracy'],
        'target_accuracy': 0.85,
        'target_met': best_model['accuracy'] >= 0.85,
        'all_metrics': all_metrics,
        'summary': {
            'avg_accuracy': np.mean([m['accuracy'] for m in all_metrics]),
            'avg_f1_score': np.mean([m['f1_score'] for m in all_metrics]),
            'avg_roc_auc': np.mean([m['roc_auc'] for m in all_metrics])
        }
    }
    
    with open('public/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[v0] Evaluation report saved")
    print(f"[v0] Best model: {best_model['model_name']} with {best_model['accuracy']:.2%} accuracy")
    print(f"[v0] Target accuracy (85%) {'MET ✓' if report['target_met'] else 'NOT MET ✗'}")
    
    return report

# Main execution
if __name__ == "__main__":
    print("[v0] Starting model evaluation pipeline...")
    
    # Load models and test data
    models = load_models()
    X_test, y_test = load_test_data()
    
    # Evaluate all models
    all_metrics = []
    for name, model in models.items():
        metrics, _, _ = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
    
    # Generate visualizations
    plot_confusion_matrices(all_metrics)
    plot_roc_curves(models, X_test, y_test)
    plot_feature_importance(models, None)
    plot_metrics_comparison(all_metrics)
    
    # Generate report
    report = generate_evaluation_report(all_metrics)
    
    print("[v0] Model evaluation complete!")
    print(f"[v0] All visualizations saved to public/plots/")
