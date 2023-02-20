import os
import uuid
from flask import Flask, request, jsonify
import whisper
import tempfile

# Flask app
app = Flask(__name__)

# Load model
model = whisper.load_model("base")

# Transcribe route
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file not found'}), 400

    # Audio file
    audio_file = request.files['audio']

    # Save audio file into tmp folder
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, str(uuid.uuid4()))
    audio_file.save(tmp_path)

    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(tmp_path)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)

    # Decode the audio
    result = whisper.transcribe(model, tmp_path)
    text_result = result["text"]
    text_result_trim = text_result.strip()

    # Delete tmp file
    os.remove(tmp_path)
    
    return jsonify({
        'language': language,
        'text': text_result_trim
    }), 200

# Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

# Entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))

    print("[Whisper REST API] Starting server on port " + str(port))

    # Start the server
    app.run(port=port, debug=False)
