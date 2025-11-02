"""
Example: Using TEN for Time Series Forecasting

This example demonstrates how to use TEN_TimeSeries for forecasting tasks
like stock prices, weather prediction, sensor data, etc.

Copyright (c) 2025 Genovo Technologies. All Rights Reserved.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model
from src.train import Trainer


class TimeSeriesDataset(Dataset):
    """Dummy time series dataset for demonstration."""
    def __init__(self, num_samples: int, seq_len: int, input_dim: int, forecast_horizon: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic time series (in practice, load real data)
        t = np.linspace(0, 4 * np.pi, self.seq_len + self.forecast_horizon)
        data = np.sin(t) + np.random.randn(len(t)) * 0.1
        data = data[:, np.newaxis].repeat(self.input_dim, axis=1)
        
        # Split into input and target
        inputs = torch.tensor(data[:self.seq_len], dtype=torch.float32)
        targets = torch.tensor(data[self.seq_len:self.seq_len + self.forecast_horizon], 
                              dtype=torch.float32)
        
        return inputs, targets


def main():
    print("=" * 80)
    print("TEN for Time Series Forecasting Example")
    print("=" * 80)
    
    # Configuration
    input_dim = 5  # e.g., 5 sensor readings or features
    forecast_horizon = 24  # Predict next 24 timesteps
    seq_len = 96  # Use 96 past timesteps
    
    # Create TEN TimeSeries model
    print("\nCreating TEN TimeSeries model...")
    model = create_model(
        model_type='timeseries',
        input_dim=input_dim,
        d_model=256,
        num_layers=4,
        num_eigenstates=64,
        forecast_horizon=forecast_horizon,
        output_dim=input_dim,  # Forecast same features
        max_seq_len=seq_len,
        dropout=0.1,
    )
    
    print(f"✓ Model created with {model.count_parameters() / 1e6:.2f}M parameters")
    print(f"  Input dim: {input_dim}")
    print(f"  Forecast horizon: {forecast_horizon}")
    print(f"  History length: {seq_len}")
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = TimeSeriesDataset(1000, seq_len, input_dim, forecast_horizon)
    val_dataset = TimeSeriesDataset(200, seq_len, input_dim, forecast_horizon)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Setup training
    print("\nSetting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        language_modeling=False,
        grad_clip=1.0,
    )
    
    # Train
    print("\nTraining...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss (MSE): {history['train_loss'][-1]:.6f}")
    print(f"  Final val loss (MSE): {history['val_loss'][-1]:.6f}")
    
    # Test inference
    print("\nTesting forecasting...")
    model.eval()
    test_input = torch.randn(1, seq_len, input_dim)
    
    with torch.no_grad():
        forecast = model(test_input)
    
    print(f"  Input shape: {test_input.shape}")
    print(f"  Forecast shape: {forecast.shape}")
    print(f"  Sample forecast values: {forecast[0, :5, 0].numpy()}")
    
    # Multi-step forecasting (recursive)
    print("\nTesting multi-step forecasting...")
    with torch.no_grad():
        current_input = test_input.clone()
        all_forecasts = []
        
        for step in range(3):  # Forecast 3 horizons ahead
            forecast = model(current_input)
            all_forecasts.append(forecast)
            
            # Update input: shift and append forecast
            current_input = torch.cat([
                current_input[:, forecast_horizon:, :],
                forecast
            ], dim=1)
    
    print(f"  Generated {len(all_forecasts)} forecast horizons")
    print(f"  Total forecast length: {len(all_forecasts) * forecast_horizon} timesteps")
    
    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
