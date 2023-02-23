import os
import re
import tempfile
import uuid
from flask import Flask, jsonify, request, send_file
from num2words import num2words
from pydub import AudioSegment
import torchaudio
from speechbrain.pretrained import HIFIGAN, Tacotron2
import whisper

# Flask app
app = Flask(__name__)

# Load TTS model
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

# TTS file prefix
speech_tts_prefix = "speech-tts-"
wav_suffix = ".wav"
opus_suffix = ".opus"

# Load transcription model
model = whisper.load_model("base")

# Clean temporary files (called every 5 minutes)
def clean_tmp():
    tmp_dir = tempfile.gettempdir()
    for file in os.listdir(tmp_dir):
        if file.startswith(speech_tts_prefix):
            os.remove(os.path.join(tmp_dir, file))
    print("[Speech REST API] Temporary files cleaned!")

# Preprocess text to replace numerals with words
def preprocess_text(text):
    text = re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
    return text

# Run TTS and save file
# Returns the path to the file
def run_tts_and_save_file(text):
    # Running the TTS
    mel_outputs, mel_length, alignment = tacotron2.encode_batch([text])

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_outputs)

    # Get temporary directory
    tmp_dir = tempfile.gettempdir()

    # Save wav to temporary file
    tmp_path_wav = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + wav_suffix)
    torchaudio.save(tmp_path_wav, waveforms.squeeze(1), 22050)
    return tmp_path_wav

# TTS endpoint
@app.route('/tts', methods=['POST'])
def generate_tts():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Invalid input: text missing'}), 400

    # Sentences to generate
    text = request.json['text']

    # Remove ' and " and  from text
    text = text.replace("'", "")
    text = text.replace('"', "")

    # Preprocess text to replace numerals with words
    text = preprocess_text(text)

    # Split text by . ? !
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

    # Trim sentences
    sentences = [sentence.strip() for sentence in sentences]

    # Remove empty sentences
    sentences = [sentence for sentence in sentences if sentence]

    # Logging
    print("[Speech REST API] Got request: length (" + str(len(text)) + "), sentences (" + str(len(sentences)) + ")")

    # Run TTS for each sentence
    output_files = []

    for sentence in sentences:
        print("[Speech REST API] Generating TTS: " + sentence)
        tmp_path_wav = run_tts_and_save_file(sentence)
        output_files.append(tmp_path_wav)

    # Concatenate all files
    audio = AudioSegment.empty()

    for file in output_files:
        audio += AudioSegment.from_wav(file)

    # Save audio to file
    tmp_dir = tempfile.gettempdir()
    tmp_path_opus = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + opus_suffix)
    audio.export(tmp_path_opus, format="opus")

    # Delete tmp files
    for file in output_files:
        os.remove(file)

    # Send file response
    return send_file(tmp_path_opus, mimetype='audio/ogg, codecs=opus')

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

@app.route('/clean', methods=['GET'])
def clean():
    clean_tmp()
    return jsonify({'status': 'ok'}), 200

# Entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))

    # Start server
    print("[Speech REST API] Starting server on port " + str(port))

    app.run(host='0.0.0.0', port=3000)