
from matplotlib import pyplot as plt


def create_loss_plot(train_loss_history, val_loss_history, params):
    plt.figure(figsize=(10, 6))

    plt.plot(train_loss_history, label='Training Loss', color='blue', linestyle='--', marker='o')

    plt.plot(val_loss_history, label='Validation Loss', color='red', linestyle='-', marker='x')

    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel(f'Config: {params}', fontsize=8)
    plt.ylabel('Loss', fontsize=14)

    plt.legend(fontsize=12)

    plt.grid(True)

    plt.savefig(f'experiments/loss_history.png', bbox_inches='tight')


def create_metric_plot(val_metric_history, params):
    plt.figure(figsize=(10, 6))

    plt.plot(val_metric_history, label='Validation roc-auc', color='red')

    plt.title('Training and Validation roc-auc(float) Over Epochs', fontsize=16)
    plt.xlabel(f'Config: {params}', fontsize=8)
    plt.ylabel('ROC-AUC', fontsize=14)

    plt.legend(fontsize=12)
    plt.grid(True)

    plt.savefig(f'experiments/metric_history.png', bbox_inches='tight')

def plot_roc_auc(fpr, tpr, roc_auc):

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Validation)')
    plt.legend(loc="lower right")
    plt.grid(True)


    plt.savefig('experiments/validation_roc_curve_.png', dpi=300, bbox_inches='tight')
