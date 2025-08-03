#!/bin/bash
# Force Python 3.9 installation
pyenv install 3.9.16 -s
pyenv global 3.9.16

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt