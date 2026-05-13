import os
import matplotlib.pyplot as plt

def plot_loss_curves(train_loss: list, valid_loss: list, save_path: str):
    """
    Plots training and validation loss curves and saves them to a file.
    
    Args:
        train_loss: List of training loss values per epoch.
        valid_loss: List of validation (test) loss values per epoch.
        save_path: File path to save the generated plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Validation / Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()
