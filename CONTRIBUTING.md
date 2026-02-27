# Contributing to VPS-FastSearch

Thanks for your interest in contributing! This document provides guidelines for contributing to VPS-FastSearch.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/NarlySoftware/VPS-fastsearch
   cd VPS-fastsearch
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev,all]"
   ```

## Development Setup

```bash
# Run unit tests
pytest tests/ -v

# Run integration/benchmark suite
python run_tests.py

# Run linting
ruff check vps_fastsearch/
black --check vps_fastsearch/

# Format code
black vps_fastsearch/
```

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests if applicable

3. Ensure tests pass and code is formatted

4. Commit with a clear message:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

5. Push and create a Pull Request

## Code Style

- Follow PEP 8 (enforced by ruff/black)
- Line length: 100 characters
- Use type hints for function signatures
- Add docstrings for public functions and classes
- Keep functions focused and reasonably sized

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system and architecture (x86_64 or aarch64)
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages or logs

## Feature Requests

Feature requests are welcome! Please describe:
- The use case
- Expected behavior
- Why existing features don't cover this

## Questions?

Open an issue with the "question" label if you need help or clarification.
