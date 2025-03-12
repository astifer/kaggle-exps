from matplotlib import pyplot as plt

def create_loss_plot(loss_history, name: str= "Loss"):
    plt.figure(figsize=(10, 6))

    plt.plot(loss_history, label=name, color='blue', linestyle='--', marker='o')


    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel(f'Epochs', fontsize=8)
    plt.ylabel('Loss', fontsize=14)

    plt.legend(fontsize=12)

    plt.grid(True)

    plt.savefig(f'plots/loss_history_{name}.png', bbox_inches='tight')

def create_metric_plot(metrc_history, name: str):
    plt.figure(figsize=(10, 6))

    plt.plot(metrc_history, label=name, color='red', linestyle='--', marker='o')


    plt.title('Metric History Over Epochs', fontsize=16)
    plt.xlabel(f'Epochs', fontsize=8)
    plt.ylabel(f'Metric {name}', fontsize=14)

    plt.legend(fontsize=12)

    plt.grid(True)

    plt.savefig(f'plots/metric_history_{name}.png', bbox_inches='tight')