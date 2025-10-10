#!/bin/bash

echo "Installing PLLM MVE dependencies..."

# Check if in conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda environment detected: $CONDA_DEFAULT_ENV"
else
    echo "Warning: No conda environment detected. Consider activating 'llm-finetune' environment first."
fi

# Install requirements
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

echo "Installation complete!"
echo "You can now run: python run_experiment.py --debug"