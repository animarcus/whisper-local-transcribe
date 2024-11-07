#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the virtual environment relative to the script's location
VENV_PATH="$SCRIPT_DIR/transcribe_venv"  # Adjust this path as needed

# Path to the transcription script relative to the script's location
SCRIPT_PATH="$SCRIPT_DIR/cli_transcribe.py"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run the transcription script with arguments
python "$SCRIPT_PATH" "$@"

# Deactivate the virtual environment
deactivate
