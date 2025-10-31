# Temporal Eigenstate Networks - Documentation

Welcome to the Temporal Eigenstate Networks (TEN) documentation.

## 📚 Documentation Index

### Getting Started
- **[Installation Guide](INSTALLATION.md)** - Complete guide for installing and packaging TEN as a module
  - Installation options (development, wheel, git)
  - Using TEN in your applications
  - Distribution within your organization
  - Docker integration
  - Troubleshooting

### Technical Documentation
- **[Architecture Evaluation](ARCHITECTURE_EVALUATION.md)** - Comprehensive technical analysis
  - Mathematical foundation (10/10)
  - Complexity analysis (O(T) vs O(T²))
  - Theoretical guarantees (universal approximation, Lyapunov stability)
  - Implementation quality assessment
  - Comparison with state-of-the-art
  - Overall score: 9.2/10 ⭐

### Security & Compliance
- **[Confidentiality Notice](CONFIDENTIALITY.md)** - Proprietary information and restrictions
  - Internal use only
  - Non-disclosure requirements
  - Restricted distribution
  
- **[Security Policy](SECURITY.md)** - Security guidelines and procedures
  - Vulnerability reporting
  - Security best practices
  - Contact information

## 📖 Main Documentation

The main README is located at the root: [`../README.md`](../README.md)

It contains:
- Overview and key features
- Quick start guide
- Architecture explanation
- Usage examples
- Benchmarks
- Paper and citation information

## 🔍 Quick Links

### Code Structure
```
temporal-eigenstate-networks/
├── README.md                     # Main documentation
├── docs/                         # Detailed documentation (you are here)
│   ├── README.md                # This file
│   ├── INSTALLATION.md          # Installation & packaging guide
│   ├── ARCHITECTURE_EVALUATION.md  # Technical evaluation
│   ├── CONFIDENTIALITY.md       # Confidentiality notice
│   └── SECURITY.md              # Security policy
├── src/                         # Source code
│   ├── model.py                # TEN architecture
│   ├── train.py                # Training utilities
│   └── eval.py                 # Evaluation utilities
├── examples/                    # Usage examples
│   ├── quickstart.ipynb        # Interactive tutorial
│   └── benchmarks.py           # Performance benchmarks
├── tests/                       # Unit tests
│   └── test_model.py           # Model tests
└── paper/                       # Research paper
    └── paper.tex               # LaTeX source
```

## 🚀 Quick Start

```python
from src import TemporalEigenstateNetwork, TemporalEigenstateConfig

# Configure model
config = TemporalEigenstateConfig(
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=2048,
)

# Create and use
model = TemporalEigenstateNetwork(config)
output = model(input_tensor)
```

## 📊 Key Performance Metrics

| Metric | Value |
|--------|-------|
| Complexity | O(T) vs O(T²) for transformers |
| Speedup | 3-28× faster (seq length 512-8192) |
| Memory Savings | 120× less memory |
| Architecture Score | 9.2/10 |
| Publication Ready | ✅ Yes |

## 🔬 Research

**Paper**: Temporal Eigenstate Networks: Linear-Complexity Sequence Modeling via Spectral Decomposition

**Author**: Oluwatosin Afolabi (Genovo Technologies)

**Status**: Ready for submission to NeurIPS/ICML/ICLR 2025

## 📧 Support

For internal support and questions:
- **Author**: Oluwatosin Afolabi
- **Email**: afolabi@genovotech.com
- **Company**: Genovo Technologies

## ⚖️ License

**Proprietary** - Internal Use Only

Copyright © 2025 Genovo Technologies. All Rights Reserved.

This is proprietary software. See [CONFIDENTIALITY.md](CONFIDENTIALITY.md) for details.

---

*Last Updated: October 31, 2025*
