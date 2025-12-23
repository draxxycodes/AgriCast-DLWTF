"""
Training Module for Agricultural Price Prediction Models

This module implements comprehensive training utilities including:
- Custom callbacks for monitoring and control
- Learning rate scheduling with warmup
- Training loops with early stopping
- Time series cross-validation
- Model checkpointing
"""

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, LearningRateScheduler, Callback
)
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    Provides smoother training:
    - Linear warmup from 0 to initial_lr
    - Cosine decay from initial_lr to min_lr
    """
    
    def __init__(
        self,
        initial_learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        min_learning_rate: float = 1e-7
    ):
        super(WarmupCosineSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_learning_rate = min_learning_rate
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        
        # Linear warmup
        warmup_lr = self.initial_learning_rate * (step / warmup_steps)
        
        # Cosine decay
        decay_progress = (step - warmup_steps) / (decay_steps - warmup_steps)
        decay_progress = tf.minimum(decay_progress, 1.0)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_progress))
        decay_lr = (self.initial_learning_rate - self.min_learning_rate) * cosine_decay + self.min_learning_rate
        
        # Choose based on step
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'min_learning_rate': self.min_learning_rate
        }


class TrainingProgressCallback(Callback):
    """
    Custom callback for detailed training progress monitoring.
    
    Features:
    - Epoch timing
    - Best metric tracking
    - Progress visualization
    """
    
    def __init__(self, model_name: str = "Model"):
        super(TrainingProgressCallback, self).__init__()
        self.model_name = model_name
        self.epoch_start_time = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.now()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = (datetime.now() - self.epoch_start_time).total_seconds()
        val_loss = logs.get('val_loss', float('inf'))
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            improvement = "★ New best!"
        else:
            improvement = ""
        
        print(f"  [{self.model_name}] Epoch {epoch+1} - "
              f"Loss: {logs.get('loss', 0):.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val MAE: {logs.get('val_mae', 0):.4f} | "
              f"Time: {elapsed:.1f}s {improvement}")
    
    def on_train_end(self, logs=None):
        print(f"\n[{self.model_name}] Training complete!")
        print(f"  Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")


class GradientMonitorCallback(Callback):
    """
    Monitor gradient statistics during training.
    
    Helps detect vanishing/exploding gradients.
    """
    
    def __init__(self, log_frequency: int = 10):
        super(GradientMonitorCallback, self).__init__()
        self.log_frequency = log_frequency
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.log_frequency == 0:
            gradients = []
            for layer in self.model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    grad = tf.reduce_mean(tf.abs(layer.kernel)).numpy()
                    gradients.append(grad)
            
            if gradients:
                avg_grad = np.mean(gradients)
                max_grad = np.max(gradients)
                
                if max_grad > 10:
                    print(f"\n⚠️ Warning: Large gradients detected (max: {max_grad:.4f})")
                elif avg_grad < 1e-7:
                    print(f"\n⚠️ Warning: Vanishing gradients detected (avg: {avg_grad:.8f})")


def get_callbacks(
    model_name: str,
    checkpoint_dir: str = 'models',
    log_dir: str = 'logs',
    patience: int = 15,
    reduce_lr_patience: int = 5,
    min_lr: float = 1e-7,
    monitor: str = 'val_loss'
) -> List[Callback]:
    """
    Get standard set of training callbacks.
    
    Args:
        model_name: Name of the model (for file saving)
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        patience: Patience for early stopping
        reduce_lr_patience: Patience for LR reduction
        min_lr: Minimum learning rate
        monitor: Metric to monitor
        
    Returns:
        List of Keras callbacks
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks_list = [
        # Early stopping
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Model checkpointing
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'{model_name}_best.keras'),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
            mode='min'
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
            mode='min'
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(log_dir, f'{model_name}_{timestamp}'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # Custom progress callback
        TrainingProgressCallback(model_name=model_name)
    ]
    
    return callbacks_list


class ModelTrainer:
    """
    Comprehensive model training utility.
    
    Features:
    - Unified training interface
    - Automatic callback setup
    - Training history tracking
    - Multi-model training support
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'models',
        log_dir: str = 'logs',
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        verbose: int = 1
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            checkpoint_dir: Directory for saving models
            log_dir: Directory for TensorBoard logs
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level
        """
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        
        self.histories = {}
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    def train(
        self,
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str = 'model',
        callbacks: List[Callback] = None,
        class_weights: Dict = None,
        **kwargs
    ) -> tf.keras.callbacks.History:
        """
        Train a single model.
        
        Args:
            model: Keras model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name for the model
            callbacks: Custom callbacks (uses defaults if None)
            class_weights: Class weights for imbalanced data
            **kwargs: Additional fit arguments
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = get_callbacks(
                model_name=model_name,
                checkpoint_dir=self.checkpoint_dir,
                log_dir=self.log_dir,
                patience=self.patience
            )
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=self.verbose,
            **kwargs
        )
        
        self.histories[model_name] = history
        
        return history
    
    def train_multiple(
        self,
        models: Dict[str, tf.keras.Model],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs
    ) -> Dict[str, tf.keras.callbacks.History]:
        """
        Train multiple models sequentially.
        
        Args:
            models: Dictionary of model name to model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary of model name to history
        """
        for name, model in models.items():
            self.train(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model_name=name,
                **kwargs
            )
        
        return self.histories
    
    def get_best_model(self) -> Tuple[str, float]:
        """
        Get the best performing model based on validation loss.
        
        Returns:
            Tuple of (model_name, best_val_loss)
        """
        best_name = None
        best_loss = float('inf')
        
        for name, history in self.histories.items():
            min_loss = min(history.history['val_loss'])
            if min_loss < best_loss:
                best_loss = min_loss
                best_name = name
        
        return best_name, best_loss
    
    def load_best_model(self, model_name: str) -> tf.keras.Model:
        """
        Load the best saved model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded Keras model
        """
        model_path = os.path.join(self.checkpoint_dir, f'{model_name}_best.keras')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return tf.keras.models.load_model(model_path)


class TimeSeriesCrossValidator:
    """
    Time series cross-validation for model evaluation.
    
    Uses expanding window approach where training data
    expands with each fold while maintaining temporal order.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        min_train_size: int = None,
        test_size: int = None
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of CV splits
            gap: Gap between train and test to prevent leakage
            min_train_size: Minimum training set size
            test_size: Size of test set for each fold
        """
        self.n_splits = n_splits
        self.gap = gap
        self.min_train_size = min_train_size
        self.test_size = test_size
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.
        
        Args:
            X: Input data
            
        Returns:
            List of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        if self.min_train_size is None:
            min_train_size = n_samples // (self.n_splits + 1)
        else:
            min_train_size = self.min_train_size
        
        splits = []
        
        for i in range(self.n_splits):
            train_end = min_train_size + i * test_size
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def cross_validate(
        self,
        model_builder,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_builder: Function that returns a compiled model
            X: Input features
            y: Targets
            epochs: Training epochs per fold
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Dictionary of metric to list of fold scores
        """
        splits = self.split(X)
        
        results = {
            'val_loss': [],
            'val_mae': [],
            'val_mse': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"\nFold {fold + 1}/{len(splits)}")
            print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            
            # Get data splits
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Build fresh model
            model = model_builder()
            
            # Train
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=verbose
            )
            
            # Evaluate
            scores = model.evaluate(X_test, y_test, verbose=0)
            
            results['val_loss'].append(scores[0])
            results['val_mae'].append(scores[1])
            results['val_mse'].append(scores[2])
            
            print(f"  Fold {fold + 1} - Loss: {scores[0]:.6f}, MAE: {scores[1]:.4f}")
            
            # Clear session to free memory
            tf.keras.backend.clear_session()
        
        # Print summary
        print(f"\n{'='*40}")
        print("Cross-Validation Results")
        print(f"{'='*40}")
        for metric, values in results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric}: {mean_val:.6f} ± {std_val:.6f}")
        
        return results


def create_learning_rate_schedule(
    initial_lr: float = 0.001,
    warmup_epochs: int = 5,
    total_epochs: int = 100,
    steps_per_epoch: int = 100,
    schedule_type: str = 'cosine'
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    Create a learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        steps_per_epoch: Steps per epoch
        schedule_type: Type of schedule ('cosine', 'exponential', 'step')
        
    Returns:
        Learning rate schedule
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    if schedule_type == 'cosine':
        return WarmupCosineSchedule(
            initial_learning_rate=initial_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps
        )
    elif schedule_type == 'exponential':
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=steps_per_epoch * 10,
            decay_rate=0.9,
            staircase=True
        )
    else:
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[warmup_steps, total_steps // 2, total_steps * 3 // 4],
            values=[initial_lr, initial_lr, initial_lr / 5, initial_lr / 25]
        )
