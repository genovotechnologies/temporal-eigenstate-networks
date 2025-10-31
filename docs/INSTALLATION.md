# Installation & Packaging Guide

## Installation Options

### Option 1: Install in Development Mode (Recommended for Internal Use)

This allows you to edit the source code and see changes immediately:

```bash
cd /path/to/temporal-eigenstate-networks
pip install -e .
```

With full dependencies (matplotlib, scipy, etc.):
```bash
pip install -e ".[full]"
```

With development tools (pytest, black, mypy):
```bash
pip install -e ".[dev]"
```

All extras:
```bash
pip install -e ".[full,dev]"
```

### Option 2: Install from Source

```bash
cd /path/to/temporal-eigenstate-networks
pip install .
```

### Option 3: Build and Install Wheel (For Distribution)

Build the package:
```bash
python -m pip install --upgrade build
python -m build
```

This creates:
- `dist/temporal_eigenstate_networks-0.1.0-py3-none-any.whl` (wheel file)
- `dist/temporal-eigenstate-networks-0.1.0.tar.gz` (source distribution)

Install the wheel:
```bash
pip install dist/temporal_eigenstate_networks-0.1.0-py3-none-any.whl
```

### Option 4: Install Directly from Git Repository

If you have the code in a private Git repository:

```bash
pip install git+https://github.com/genovotechnologies/temporal-eigenstate-networks.git
```

Or for a specific branch/tag:
```bash
pip install git+https://github.com/genovotechnologies/temporal-eigenstate-networks.git@main
```

---

## Using TEN in Your Application

### Basic Import

After installation, you can import TEN in any Python project:

```python
# Import the core components
from src import TemporalEigenstateNetwork, TemporalEigenstateConfig

# Or import specific modules
from src.model import TemporalEigenstateNetwork, TemporalEigenstateConfig
from src.train import Trainer
from src.eval import Evaluator
```

### Example Application Integration

```python
"""
Your Application: my_app/models/sequence_processor.py
"""
import torch
from src import TemporalEigenstateNetwork, TemporalEigenstateConfig

class MySequenceProcessor:
    def __init__(self, input_dim: int, output_dim: int):
        # Configure TEN
        self.config = TemporalEigenstateConfig(
            d_model=512,
            n_heads=8,
            n_layers=6,
            max_seq_len=2048,
            num_eigenstates=64,
        )
        
        # Create the model
        self.ten_model = TemporalEigenstateNetwork(self.config)
        
        # Add your custom layers
        self.input_projection = torch.nn.Linear(input_dim, self.config.d_model)
        self.output_projection = torch.nn.Linear(self.config.d_model, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.ten_model(x)
        x = self.output_projection(x)
        return x

# Use in your application
processor = MySequenceProcessor(input_dim=128, output_dim=10)
data = torch.randn(4, 100, 128)  # batch_size=4, seq_len=100
output = processor.forward(data)
```

### Integration with Existing Applications

#### Example 1: As a Drop-in Replacement for Transformer

```python
from src import TemporalEigenstateNetwork, TemporalEigenstateConfig

class MyExistingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # OLD: self.transformer = torch.nn.Transformer(...)
        
        # NEW: Replace with TEN
        config = TemporalEigenstateConfig(
            d_model=512,
            n_heads=8,
            n_layers=6,
        )
        self.encoder = TemporalEigenstateNetwork(config)
        
    def forward(self, x):
        return self.encoder(x)
```

#### Example 2: Hybrid Architecture

```python
from src import TemporalEigenstateNetwork, TemporalEigenstateConfig

class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use TEN for long-range dependencies
        ten_config = TemporalEigenstateConfig(
            d_model=512,
            n_heads=8,
            n_layers=4,
            num_eigenstates=64,
        )
        self.temporal_encoder = TemporalEigenstateNetwork(ten_config)
        
        # Use other components for specific tasks
        self.cnn_feature_extractor = torch.nn.Conv1d(...)
        self.final_classifier = torch.nn.Linear(...)
        
    def forward(self, x):
        # Extract features
        features = self.cnn_feature_extractor(x)
        
        # Process with TEN for temporal modeling
        temporal_features = self.temporal_encoder(features)
        
        # Final prediction
        output = self.final_classifier(temporal_features)
        return output
```

#### Example 3: Fine-tuning for Your Domain

```python
from src import TemporalEigenstateNetwork, TemporalEigenstateConfig
from src.train import Trainer

# Create model
config = TemporalEigenstateConfig(
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=1024,
)
model = TemporalEigenstateNetwork(config)

# Load pre-trained weights (if available)
# model.load_state_dict(torch.load('pretrained_ten.pt'))

# Add task-specific head
task_head = torch.nn.Sequential(
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, num_classes),
)

# Fine-tune on your data
class MyTaskModel(torch.nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
        
    def forward(self, x):
        features = self.encoder(x)
        return self.head(features[:, -1, :])  # Use last token

my_model = MyTaskModel(model, task_head)

# Train using the provided Trainer
optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
trainer = Trainer(my_model, optimizer, criterion)

trainer.fit(train_loader, val_loader, epochs=10)
```

---

## Distributing Within Your Organization

### Option 1: Private PyPI Server

Set up a private PyPI server (e.g., using `devpi` or `pypiserver`):

```bash
# Build the package
python -m build

# Upload to your private PyPI
twine upload --repository-url https://your-private-pypi.genovotech.com dist/*
```

Then install from your private server:
```bash
pip install --index-url https://your-private-pypi.genovotech.com temporal-eigenstate-networks
```

### Option 2: Shared Network Drive

Build and share the wheel file:

```bash
# Build
python -m build

# Copy to shared drive
cp dist/temporal_eigenstate_networks-0.1.0-py3-none-any.whl /shared/packages/
```

Install from shared drive:
```bash
pip install /shared/packages/temporal_eigenstate_networks-0.1.0-py3-none-any.whl
```

### Option 3: Git Repository (Private)

Developers can install directly:

```bash
# Via SSH
pip install git+ssh://git@github.com/genovotechnologies/temporal-eigenstate-networks.git

# Via HTTPS with token
pip install git+https://token@github.com/genovotechnologies/temporal-eigenstate-networks.git
```

---

## Version Management

### Updating Version

Edit `src/__init__.py`:

```python
__version__ = "0.2.0"  # Update this
```

Then rebuild:
```bash
python -m build
```

### Using in requirements.txt

```
# requirements.txt for your application

# Option 1: From wheel
temporal-eigenstate-networks @ file:///path/to/temporal_eigenstate_networks-0.1.0-py3-none-any.whl

# Option 2: From Git
temporal-eigenstate-networks @ git+https://github.com/genovotechnologies/temporal-eigenstate-networks.git@v0.1.0

# Option 3: Editable install (development)
-e /path/to/temporal-eigenstate-networks
```

### Using in pyproject.toml

```toml
[project]
dependencies = [
    "temporal-eigenstate-networks @ file:///path/to/dist/temporal_eigenstate_networks-0.1.0-py3-none-any.whl",
]
```

---

## Docker Integration

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy your application
COPY . /app

# Install TEN from wheel (place wheel in docker context)
COPY temporal_eigenstate_networks-0.1.0-py3-none-any.whl /tmp/
RUN pip install /tmp/temporal_eigenstate_networks-0.1.0-py3-none-any.whl

# Or install from Git
# RUN pip install git+https://github.com/genovotechnologies/temporal-eigenstate-networks.git

# Install your app
RUN pip install -r requirements.txt

CMD ["python", "your_app.py"]
```

---

## Testing Your Installation

```python
# test_installation.py
import sys

try:
    from src import TemporalEigenstateNetwork, TemporalEigenstateConfig
    print("✓ Successfully imported TEN")
    
    # Test basic functionality
    import torch
    config = TemporalEigenstateConfig(d_model=256, n_heads=4, n_layers=2)
    model = TemporalEigenstateNetwork(config)
    
    x = torch.randn(2, 10, 256)
    output = model(x)
    
    assert output.shape == x.shape, "Output shape mismatch"
    print("✓ Basic forward pass successful")
    print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\n✅ Installation verified successfully!")
    
except Exception as e:
    print(f"❌ Installation verification failed: {e}")
    sys.exit(1)
```

Run the test:
```bash
python test_installation.py
```

---

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'src'`:

1. Check installation:
   ```bash
   pip list | grep temporal-eigenstate
   ```

2. Reinstall in editable mode:
   ```bash
   pip install -e .
   ```

3. Verify package is in sys.path:
   ```python
   import sys
   print(sys.path)
   ```

### PyTorch Version Conflicts

TEN requires PyTorch >= 2.0.0. If you have an older version:

```bash
pip install --upgrade torch>=2.0.0
```

### CUDA Issues

For GPU support, install PyTorch with CUDA:

```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## Best Practices

1. **Pin Versions**: In production, pin exact versions
   ```
   temporal-eigenstate-networks==0.1.0
   ```

2. **Use Virtual Environments**: Always use venv or conda
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

3. **Cache Builds**: Cache wheels for faster installation
   ```bash
   pip wheel . -w ./wheels
   pip install ./wheels/temporal_eigenstate_networks-0.1.0-py3-none-any.whl
   ```

4. **Document Dependencies**: Keep requirements.txt updated
   ```bash
   pip freeze > requirements.txt
   ```

---

## Support

For internal support:
- **Email**: afolabi@genovotech.com
- **Internal Wiki**: [Link to your internal documentation]
- **Slack**: #ten-support (or your internal channel)

---

## License Reminder

This is proprietary software owned by Genovo Technologies. Do not distribute outside the organization. See LICENSE and CONFIDENTIALITY.md for details.
