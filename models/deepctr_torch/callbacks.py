import torch
import operator
from typing import Dict, Any, Optional


class History:
    """Minimal History callback to store metrics per epoch."""

    def __init__(self):
        self.history: Dict[str, list] = {}

    def on_train_begin(self):
        self.history = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class EarlyStopping:
    """Simple EarlyStopping implementation without TensorFlow.

    Parameters:
        monitor: metric name to monitor, e.g., 'val_loss' or 'val_auc'
        min_delta: minimum change to qualify as an improvement
        patience: epochs to wait after last improvement
        mode: 'min' or 'max' to indicate direction of improvement
        verbose: print messages when improvement occurs
    """

    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0.0, patience: int = 0,
                 mode: str = 'min', verbose: int = 0):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.best: Optional[float] = None
        self.wait = 0
        self.stop_training = False
        self.operator = operator.lt if mode == 'min' else operator.gt

    def on_train_begin(self):
        self.best = None
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        # adjust by min_delta per mode
        threshold = (current + self.min_delta) if self.mode == 'min' else (current - self.min_delta)

        if self.best is None or self.operator(threshold, self.best):
            # if self.verbose:
            #     print(f"Epoch {epoch + 1:05d}: {self.monitor} improved from {self.best} to {current}")
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                if self.verbose:
                    print(f"Epoch {epoch + 1:05d}: early stopping")
                self.stop_training = True


class ModelCheckpoint:
    """Save the model or weights based on monitored metric.

    Parameters:
        filepath: path template to save to (can use {epoch} and keys from logs)
        monitor: metric to monitor
        verbose: 0/1
        save_best_only: only save when monitored metric improves
        save_weights_only: save only state_dict when True
        mode: 'min' or 'max'
        period: save every N epochs
    """

    def __init__(self, filepath: str, monitor: str = 'val_loss', verbose: int = 0,
                 save_best_only: bool = False, save_weights_only: bool = False,
                 mode: str = 'min', period: int = 1):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.period = max(1, int(period))
        self.epochs_since_last_save = 0
        self.best: Optional[float] = None
        self.monitor_op = operator.lt if mode == 'min' else operator.gt
        self.model = None  # will be set externally if needed

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if self.model is None:
            return
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0

        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                if self.verbose:
                    print(f'Can save best model only with {self.monitor} available, skipping.')
                return
            if self.best is None or self.monitor_op(current, self.best):
                # if self.verbose:
                #     print(f'Epoch {epoch + 1:05d}: {self.monitor} improved from {self.best} to {current}, saving to {filepath}')
                self.best = current
                if self.save_weights_only:
                    torch.save(self.model.state_dict(), filepath)
                else:
                    torch.save(self.model, filepath)
            else:
                # if self.verbose:
                #     print(f'Epoch {epoch + 1:05d}: {self.monitor} did not improve from {self.best}')
                pass
        else:
            if self.verbose:
                print(f'Epoch {epoch + 1:05d}: saving model to {filepath}')
            if self.save_weights_only:
                torch.save(self.model.state_dict(), filepath)
            else:
                torch.save(self.model, filepath)
