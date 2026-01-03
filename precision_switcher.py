# precision_switcher.py
import torch

class ConvergencePrecisionSwitcher:
    """
    Simple switching from float64 -> float32 once loss plateaus.
    """

    def __init__(self, model, optimizer, tol=1e-6, patience=50):
        self.model = model
        self.optimizer = optimizer
        self.tol = tol
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.current_precision = 'float64'

    def step(self, loss):
        # Check if loss improved significantly
        if loss + self.tol < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        # Switch precision if patience exceeded
        if self.counter >= self.patience and self.current_precision == 'float64':
            print("Switching model to float32 precision!")
            self.model.float()  # convert all params to float32
            self.current_precision = 'float32'
            self.counter = 0  # reset counter
        return self.current_precision
