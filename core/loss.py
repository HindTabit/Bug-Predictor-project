# core/loss.py
import numpy as np

def focal_loss(y_true, y_pred_proba, gamma=2.0, alpha=0.75):
    """
    Focal Loss – très efficace sur datasets déséquilibrés (bugs = minorité)
    Utile si tu veux ré-entraîner plus tard avec une perte personnalisée
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    p_t = y_true * y_pred_proba + (1 - y_true) * (1 - y_pred_proba)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    loss = -alpha_t * (1 - p_t)**gamma * np.log(p_t + 1e-8)
    return loss.mean()

# Bonus : fonction pour afficher l’évolution si tu fais du fine-tuning plus tard
def log_loss_history(history, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Évolution de la perte pendant l’entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()