# Speech REST API

This project provides a simple Flask API to transcribe speech from an audio file using the Whisper speech recognition library.
The API loads a pre-trained deep learning model to detect the spoken language and transcribe the speech to text.

It also provides an endpoint to generate speech from a list of input sentences using the Tacotron2 and HiFiGAN models.

## Requirements

- Python 3.9 or later

## Installation

```bash
# Clone the repository
git clone https://github.com/askrella/speech-rest-api.git

# Navigate to the project directory
cd speech-rest-api

# Install ffmpeg (Ubuntu & Debian)
sudo apt update && sudo apt install ffmpeg -y

# Install the dependencies
pip install -r requirements.txt

# (Optional) Set PORT environment variable
export PORT=3000

# Run the REST API
python app.py
```

## Documentation

This endpoint generates speech from a list of input sentences using the Tacotron2 and HiFiGAN models.

```bash
POST http://localhost:80/tts
```

The request body must be a JSON object with the following field:

- sentences (required): A list of sentences to be converted to speech.

Example:

```json
{
    "sentences": [
        "Hello, how are you?",
        "My name is ChatGPT.",
        "It's a pleasure to meet you."
    ]
}
```


The response is an audio file in WAV format containing the generated speech.


Here's an example curl command that generates speech from a list of input sentences:

```sh
curl -X POST \
  -H "Content-Type: application/json" \
  --data '{"sentences": ["Hello, how are you?", "My name is Shubh.", "It\'s a pleasure to meet you."]}' \
  http://localhost:80/tts \
  --output output.wav
```

This curl command sends a JSON payload with a list of three input sentences to the /tts endpoint, and saves the resulting audio file to output.wav.


This endpoint transcribes audio files using the Whisper model.


```bash
POST http://localhost:80/transcribe
```


The request body must be a form data object containing an audio file.

Example:

```form
audio=@path/to/your/audio/file.wav
```


The response is a JSON object containing the detected language and the transcribed text.

Example:

```json
{
    "language": "en-US",
    "text": "Hello, how are you?"
}
```


Here's an example curl command that transcribes an audio file:

```css 
curl -X POST \
  -F "audio=@path/to/your/audio/file.wav" \
  http://localhost:80/transcribe
```

This curl command sends an HTTP POST request to the /transcribe endpoint with a form data containing the audio file. The detected language and transcribed text are returned as a JSON object.
