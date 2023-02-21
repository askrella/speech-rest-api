import os
import uuid
import tempfile
from flask import Flask, request, jsonify, send_file
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN
import whisper

app = Flask(__name__)

# Load TTS model
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

# Load transcription model
model = whisper.load_model("base")

@app.route('/tts', methods=['POST'])
def generate_tts():
    if not request.json or not 'sentences' in request.json:
        return jsonify({'error': 'Invalid input'})
    sentences = request.json['sentences']
    # Running the TTS
    mel_outputs, mel_lengths, alignments = tacotron2.encode_batch(sentences)

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_outputs)

    # Save the waveform to a file
    file_path = 'example_TTS.wav'
    torchaudio.save(file_path, waveforms.squeeze(1), 22050)

    return send_file(file_path, mimetype='audio/wav')

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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))

    print("[Whisper REST API] Starting server on port " + str(port))

    app.run(host='0.0.0.0', port=80)
