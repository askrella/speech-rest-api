# Whisper REST API

This project provides a simple Flask API to transcribe speech from an audio file using the Whisper speech recognition library. The API loads a pre-trained deep learning model to detect the spoken language and transcribe the speech to text.

## Requirements

- Python 3.9 or later

## Installation

```bash
# Clone the repository
git clone https://github.com/askrella/whisper-rest-api.git

# Navigate to the project directory
cd whisper-rest-api

# Install ffmpeg (Ubuntu & Debian)
sudo apt update && sudo apt install ffmpeg -y

# Install the dependencies
pip install -r requirements.txt

# (Optional) Set PORT environment variable
export PORT=3000

# Set FLASK production environment variable
export FLASK_DEBUG=false

# Run the REST API
python app.py
```

## Documentation

### Request
```
Endpoint: /transcribe
Method: POST
Content-Type: multipart/form-data

Form Data:
    audio: audio file
```

### Response
    
```json
{
    "language": "en",
    "text": "hello world"
}
```

## Example CURL

```bash
curl -X POST -H "Content-Type: multipart/form-data" -F "audio=/path/to/audio.wav" http://localhost:3000/transcribe
```
