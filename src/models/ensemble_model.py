"""
Ensemble Model for Agricultural Price Prediction

This module implements an ensemble learning approach that combines
predictions from LSTM, GRU, and Transformer models using:
- Weighted averaging
- Stacking with meta-learner
- Dynamic weight adjustment based on recent performance
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate,
    BatchNormalization, LayerNormalization
)
import numpy as np
from typing import List, Dict, Tuple, Optional

from .lstm_model import LSTMAttentionModel
from .gru_model import BidirectionalGRUModel
from .transformer_model import TransformerModel


class StackingMetaLearner(layers.Layer):
    """
    Meta-learner for stacking ensemble predictions.
    
    Takes predictions from base models and learns optimal
    combination weights.
    """
    
    def __init__(
        self,
        hidden_units: Tuple[int, ...] = (32, 16),
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super(StackingMetaLearner, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.dense_layers = []
        self.dropout_layers = []
        self.norm_layers = []
        
        for units in self.hidden_units:
            self.dense_layers.append(Dense(units, activation='relu'))
            self.dropout_layers.append(Dropout(self.dropout_rate))
            self.norm_layers.append(BatchNormalization())
        
        self.output_layer = Dense(1, activation='linear')
        
        super(StackingMetaLearner, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = inputs
        for dense, dropout, norm in zip(self.dense_layers, self.dropout_layers, self.norm_layers):
            x = dense(x)
            x = norm(x, training=training)
            x = dropout(x, training=training)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(StackingMetaLearner, self).get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate
        })
        return config


class EnsembleModel:
    """
    Ensemble model combining LSTM, GRU, and Transformer predictions.
    
    Ensemble Methods:
    1. Simple Averaging: Equal weights for all models
    2. Weighted Averaging: Static weights based on validation performance
    3. Stacking: Meta-learner trained on base model predictions
    4. Dynamic Weighting: Weights adjusted based on recent performance
    
    Features:
    - Trains multiple base models
    - Combines predictions for improved accuracy
    - Reduces overfitting through model diversity
    - Provides uncertainty estimates
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 50,
        include_lstm: bool = True,
        include_gru: bool = True,
        include_transformer: bool = True,
        ensemble_method: str = 'stacking',
        meta_hidden_units: Tuple[int, ...] = (32, 16),
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize Ensemble model.
        
        Args:
            sequence_length: Number of time steps in input
            n_features: Number of input features
            include_lstm: Whether to include LSTM model
            include_gru: Whether to include GRU model
            include_transformer: Whether to include Transformer model
            ensemble_method: Method for combining predictions
            meta_hidden_units: Hidden units for meta-learner
            dropout_rate: Dropout probability
            learning_rate: Learning rate for meta-learner
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.include_lstm = include_lstm
        self.include_gru = include_gru
        self.include_transformer = include_transformer
        self.ensemble_method = ensemble_method
        self.meta_hidden_units = meta_hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.base_models = {}
        self.meta_model = None
        self.weights = None
        self.model = None
        
    def build_base_models(self) -> Dict[str, Model]:
        """
        Build all base models for the ensemble.
        
        Returns:
            Dictionary of model name to compiled model
        """
        if self.include_lstm:
            lstm_builder = LSTMAttentionModel(
                sequence_length=self.sequence_length,
                n_features=self.n_features
            )
            self.base_models['lstm'] = lstm_builder.build_model()
            print("✓ Built LSTM with Attention model")
        
        if self.include_gru:
            gru_builder = BidirectionalGRUModel(
                sequence_length=self.sequence_length,
                n_features=self.n_features
            )
            self.base_models['gru'] = gru_builder.build_model()
            print("✓ Built Bidirectional GRU model")
        
        if self.include_transformer:
            transformer_builder = TransformerModel(
                sequence_length=self.sequence_length,
                n_features=self.n_features
            )
            self.base_models['transformer'] = transformer_builder.build_model()
            print("✓ Built Transformer model")
        
        return self.base_models
    
    def build_stacking_model(self) -> Model:
        """
        Build stacking ensemble model with meta-learner.
        
        Returns:
            Compiled stacking model
        """
        n_base_models = len(self.base_models)
        
        # Input for stacked predictions
        stacked_input = Input(shape=(n_base_models,), name='stacked_predictions')
        
        # Meta-learner
        x = Dense(32, activation='relu')(stacked_input)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(self.dropout_rate / 2)(x)
        output = Dense(1, activation='linear', name='output')(x)
        
        meta_model = Model(inputs=stacked_input, outputs=output, name='Meta_Learner')
        
        meta_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        self.meta_model = meta_model
        return meta_model
    
    def build_end_to_end_model(self) -> Model:
        """
        Build end-to-end ensemble model (for inference).
        
        Returns:
            Compiled end-to-end ensemble model
        """
        # Shared input
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # Get predictions from each base model
        predictions = []
        
        if 'lstm' in self.base_models:
            lstm_pred = self.base_models['lstm'](inputs)
            predictions.append(lstm_pred)
        
        if 'gru' in self.base_models:
            gru_pred = self.base_models['gru'](inputs)
            predictions.append(gru_pred)
        
        if 'transformer' in self.base_models:
            transformer_pred = self.base_models['transformer'](inputs)
            predictions.append(transformer_pred)
        
        # Stack predictions
        if len(predictions) > 1:
            stacked = Concatenate(name='stacked_predictions')(predictions)
        else:
            stacked = predictions[0]
        
        # Meta-learner
        x = Dense(32, activation='relu', name='meta_dense_1')(stacked)
        x = BatchNormalization(name='meta_bn_1')(x)
        x = Dropout(self.dropout_rate, name='meta_dropout_1')(x)
        x = Dense(16, activation='relu', name='meta_dense_2')(x)
        x = Dropout(self.dropout_rate / 2, name='meta_dropout_2')(x)
        output = Dense(1, activation='linear', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=output, name='Ensemble')
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return self.model
    
    def train_base_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: List = None
    ) -> Dict[str, tf.keras.callbacks.History]:
        """
        Train all base models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            
        Returns:
            Dictionary of model name to training history
        """
        histories = {}
        
        for name, model in self.base_models.items():
            print(f"\n{'='*50}")
            print(f"Training {name.upper()} model...")
            print(f"{'='*50}")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            histories[name] = history
            print(f"✓ {name.upper()} training complete")
        
        return histories
    
    def get_base_predictions(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from all base models.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of model name to predictions
        """
        predictions = {}
        
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X, verbose=0).flatten()
        
        return predictions
    
    def stack_predictions(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Stack predictions from base models.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Stacked prediction array
        """
        pred_list = [predictions[name] for name in sorted(predictions.keys())]
        return np.column_stack(pred_list)
    
    def predict_weighted_average(
        self,
        X: np.ndarray,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Predict using weighted average of base models.
        
        Args:
            X: Input features
            weights: Dictionary of model weights
            
        Returns:
            Weighted average predictions
        """
        if weights is None:
            weights = {name: 1.0 / len(self.base_models) for name in self.base_models}
        
        predictions = self.get_base_predictions(X)
        
        weighted_sum = np.zeros(len(X))
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = weights.get(name, 1.0 / len(self.base_models))
            weighted_sum += weight * pred
            total_weight += weight
        
        return weighted_sum / total_weight
    
    def predict_stacking(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict using stacking meta-learner.
        
        Args:
            X: Input features
            
        Returns:
            Meta-learner predictions
        """
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained. Call train_meta_learner first.")
        
        predictions = self.get_base_predictions(X)
        stacked = self.stack_predictions(predictions)
        
        return self.meta_model.predict(stacked, verbose=0).flatten()
    
    def train_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ) -> tf.keras.callbacks.History:
        """
        Train the meta-learner on base model predictions.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        # Get base model predictions on training data
        train_predictions = self.get_base_predictions(X_train)
        stacked_train = self.stack_predictions(train_predictions)
        
        # Get base model predictions on validation data
        val_predictions = self.get_base_predictions(X_val)
        stacked_val = self.stack_predictions(val_predictions)
        
        # Build and train meta-learner
        if self.meta_model is None:
            self.build_stacking_model()
        
        print("\nTraining Meta-Learner...")
        history = self.meta_model.fit(
            stacked_train, y_train,
            validation_data=(stacked_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def calculate_validation_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate model weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of model weights
        """
        predictions = self.get_base_predictions(X_val)
        
        errors = {}
        for name, pred in predictions.items():
            mse = np.mean((pred - y_val) ** 2)
            errors[name] = mse
        
        # Convert errors to weights (inverse weighting)
        total_inv_error = sum(1.0 / e for e in errors.values())
        weights = {name: (1.0 / e) / total_inv_error for name, e in errors.items()}
        
        self.weights = weights
        
        print("\nCalculated model weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")
        
        return weights
    
    def predict(
        self,
        X: np.ndarray,
        method: str = None
    ) -> np.ndarray:
        """
        Make predictions using the specified ensemble method.
        
        Args:
            X: Input features
            method: Ensemble method (defaults to init method)
            
        Returns:
            Ensemble predictions
        """
        method = method or self.ensemble_method
        
        if method == 'stacking':
            return self.predict_stacking(X)
        elif method == 'weighted':
            return self.predict_weighted_average(X, self.weights)
        elif method == 'average':
            return self.predict_weighted_average(X)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def summary(self):
        """Print summaries of all models."""
        for name, model in self.base_models.items():
            print(f"\n{'='*50}")
            print(f"{name.upper()} Model Summary")
            print(f"{'='*50}")
            model.summary()
        
        if self.meta_model:
            print(f"\n{'='*50}")
            print("Meta-Learner Summary")
            print(f"{'='*50}")
            self.meta_model.summary()
    
    def get_model(self) -> Model:
        """Get the ensemble model."""
        if self.model is None:
            self.build_end_to_end_model()
        return self.model
    
    @staticmethod
    def get_model_description() -> dict:
        """
        Get detailed model description for documentation.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': 'Ensemble Model (LSTM + GRU + Transformer)',
            'architecture': [
                'Base Model 1: LSTM with Multi-Head Attention',
                'Base Model 2: Bidirectional GRU with Residual',
                'Base Model 3: Transformer with Positional Encoding',
                'Stacking: Concatenate base predictions',
                'Meta-Learner: Dense (32) → Dense (16) → Output (1)'
            ],
            'ensemble_methods': [
                'Simple Averaging: Equal weights',
                'Weighted Averaging: Performance-based weights',
                'Stacking: Meta-learner trained on predictions',
                'Dynamic: Weights adjusted on recent performance'
            ],
            'advantages': [
                'Combines strengths of multiple architectures',
                'Reduces overfitting through diversity',
                'More robust predictions',
                'Uncertainty estimation from model disagreement',
                'Typically outperforms individual models'
            ],
            'disadvantages': [
                'Higher computational cost (trains multiple models)',
                'Increased inference time',
                'More complex to tune and maintain',
                'Requires more memory'
            ],
            'best_for': [
                'Production systems requiring reliability',
                'When accuracy is paramount',
                'Complex data with diverse patterns',
                'When uncertainty quantification is needed'
            ]
        }


def create_ensemble_model(
    sequence_length: int = 30,
    n_features: int = 50,
    **kwargs
) -> EnsembleModel:
    """
    Factory function to create Ensemble model.
    
    Args:
        sequence_length: Input sequence length
        n_features: Number of features
        **kwargs: Additional model parameters
        
    Returns:
        EnsembleModel instance
    """
    ensemble = EnsembleModel(
        sequence_length=sequence_length,
        n_features=n_features,
        **kwargs
    )
    ensemble.build_base_models()
    return ensemble
