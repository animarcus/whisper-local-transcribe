# Local Transcribe with Whisper

## Introduction

Local Transcribe with Whisper is a user-friendly desktop application that allows you to transcribe audio and video files using the Whisper ASR system. This application provides a graphical user interface (GUI) built with Python and the Tkinter library, making it easy to use even for those not familiar with programming.

## Usage on this fork

This fork was mainly to make use this project as a command line utility. I am unsure if the GUI still works, and I am developing on Mac so there are no guarantees for this to work on other systems like the original project.
Other notable additions are that the script will utilize GPU acceleration on Silicon macs.

## Installation

1. Clone this repository:

    ```bash
    git clone [your-repository-url]
    cd [repository-name]
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv transcribe_venv
    source transcribe_venv/bin/activate # On Unix/macOS
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Command Line Usage

The script can be used to transcribe audio and video files using Whisper. It supports the following file formats:

- Audio: .wav, .mp3
- Video: .mp4

### Basic Usage

```bash
./transcribe.sh [directory] --model [model-name] [options]
```

### Available Models

Multilingual Models:

- `openai/whisper-tiny`: Fastest, least accurate (39M parameters)
- `openai/whisper-base`: Fast, good for short content (74M parameters)
- `openai/whisper-small`: Balanced speed/accuracy (244M parameters)
- `openai/whisper-medium`: High accuracy, recommended default (769M parameters)
- `openai/whisper-large`: Highest accuracy (1550M parameters)
- `openai/whisper-large-v2`: Latest version, best performance (1550M parameters)

English-only Models (optimized for English, faster):

- `openai/whisper-tiny.en`
- `openai/whisper-base.en`
- `openai/whisper-small.en`
- `openai/whisper-medium.en`
- `openai/whisper-large.en`
- `openai/whisper-large-v2.en`

Choose English-only models for English transcription as they are smaller and faster than their multilingual counterparts.

### Command Line Arguments

- `directory`: Path to the directory containing audio/video files (use `.` for current directory)
- `--model`: Whisper model to use (e.g., "openai/whisper-medium")
- `--verbose`: Show detailed progress and information
- `--language`: Specify language for transcription (default: "en")

### Examples

```bash
# Transcribe files in current directory using medium model
./transcribe.sh . --model openai/whisper-medium --verbose

# Transcribe files in specific directory with language specification
./transcribe.sh /path/to/files --model openai/whisper-small --language fr
```

### Setting up an Alias (Optional)

To use the script from anywhere, you can set up an alias in your shell configuration file (e.g., .bashrc, .zshrc):

```bash
# Add this line to your ~/.bashrc or ~/.zshrc
alias whisper-transcribe='/path/to/your/transcribe.sh'
```

After setting up the alias, you can use the script from any directory:

```bash
whisper-transcribe . --model openai/whisper-medium --verbose
```

## Output

- Transcriptions are saved in a `transcriptions` subdirectory within the input directory
- Each transcription file is named `[original-filename]-transcription.txt`
- The transcription includes timestamps and corresponding text

## Features

- Supports multiple audio and video formats
- Automatic language detection (or manual specification)
- GPU acceleration when available
- Progress tracking and detailed logging in verbose mode
- Skips already transcribed files
- Includes timing and performance statistics
