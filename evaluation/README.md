# Evaluation Scripts and Tools

This directory contains evaluation scripts and utilities for analyzing model performance, comparing different approaches, and generating publication-quality figures.

## Overview

The evaluation system supports two main workflows:

1. **Automatic evaluation during training/inference**: The `evaluate.py` script runs automatically after training or when loading pre-trained models, generating metrics and visualizations.

2. **Post-hoc analysis and comparison**: The standalone evaluation scripts in this directory (`evaluate_*.py`) allow you to compare multiple models and generate comparative analyses. It assumes that you already have the samples (i.e., the `evaluate.py` script has already been run). The paths are coded for our specific runs, so you might need to modify them and adapt them to your own needs.

## Further Reading

- See [INFERENCE.md](../INFERENCE.md) for details on running inference
- See [TRAIN.md](../TRAIN.md) for training configurations
- Check the main [README.md](../README.md) for overall project overview

