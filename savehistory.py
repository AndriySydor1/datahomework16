import numpy as np

# Історія навчання моделі Частини 1
history_conv = {
    "accuracy": [0.8, 0.85, 0.9],
    "val_accuracy": [0.78, 0.82, 0.88],
    "loss": [0.5, 0.4, 0.3],
    "val_loss": [0.55, 0.45, 0.35]
}
np.save("history_conv.npy", history_conv)

# Історія навчання моделі Частини 2
history_vgg16 = {
    "accuracy": [0.85, 0.87, 0.9],
    "val_accuracy": [0.83, 0.85, 0.89],
    "loss": [0.4, 0.35, 0.3],
    "val_loss": [0.45, 0.4, 0.35]
}
np.save("history_vgg16.npy", history_vgg16)

print("Історії навчання збережено у файли history_conv.npy та history_vgg16.npy")
