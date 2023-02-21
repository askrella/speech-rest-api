import os
import uuid
import tempfile
from flask import Flask, request, jsonify, send_file
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN
import whisper
from pydub import AudioSegment

# Flask app
app = Flask(__name__)

# Load TTS model
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

# TTS file prefix
speech_tts_prefix = "speech-tts-"
wav_suffix = ".wav"
ogg_suffix = ".ogg"

# Load transcription model
model = whisper.load_model("base")

# Clean temporary files (called every 5 minutes)
def clean_tmp():
    print("[Speech REST API] Cleaning temporary files")
    tmp_dir = tempfile.gettempdir()
    for file in os.listdir(tmp_dir):
        if file.startswith(speech_tts_prefix):
            os.remove(os.path.join(tmp_dir, file))
    print("[Speech REST API] Temporary files cleaned!")

# TTS endpoint
@app.route('/tts', methods=['POST'])
def generate_tts():
    if not request.json or 'sentences' not in request.json:
        return jsonify({'error': 'Invalid input: sentences missing'}), 400

    # Sentences to generate
    sentences = request.json['sentences']

    # Running the TTS
    mel_outputs, mel_length, alignment = tacotron2.encode_batch(sentences)

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_outputs)

    # Get temporary directory
    tmp_dir = tempfile.gettempdir()

    # Save wav to temporary file
    tmp_path_wav = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + wav_suffix)
    torchaudio.save(tmp_path_wav, waveforms.squeeze(1), 22050)

    # Convert file from wav to ogg
    audio = AudioSegment.from_wav(tmp_path_wav)
    tmp_path_ogg = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + ogg_suffix)
    audio.export(tmp_path_ogg, format="ogg")

    # Delete wav file
    os.remove(tmp_path_wav)

    # Send file response
    return send_file(tmp_path_ogg, mimetype='audio/wav')

# Transcribe endpoint
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'Invalid input, form-data: audio'}), 400

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

# Health endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

# Entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))

    # TODO: Run clean_tmp() every 5 minutes

    # Start server
    print("[Speech REST API] Starting server on port " + str(port))

    app.run(host='0.0.0.0', port=3000)
