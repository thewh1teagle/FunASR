"""
wget https://github.com/thewh1teagle/vibe/raw/refs/heads/main/samples/short.wav -O audio.wav
uv run infer.py
"""
import soundfile as sf
import numpy as np
import onnxruntime as ort

# Function to read WAV file
def read_wav_file(file_path):
    audio_samples, sample_rate = sf.read(file_path, dtype='float32')
    return audio_samples, sample_rate

# Preprocess audio to truncate or pad to the model's expected length
def preprocess_audio(audio_samples, target_length):
    if len(audio_samples) > target_length:
        audio_samples = audio_samples[:target_length]
    elif len(audio_samples) < target_length:
        padding = target_length - len(audio_samples)
        audio_samples = np.pad(audio_samples, (0, padding), mode='constant')
    return audio_samples

# Perform inference with the ONNX model
def infer_emotion(model_path, audio_samples):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data = np.expand_dims(audio_samples, axis=0).astype(np.float32)
    result = session.run([output_name], {input_name: input_data})
    return result

# Main workflow
wav_file_path = "audio.wav"
onnx_model_path = "emotion2vec.onnx"
expected_length = int(16000 * 1)

# Read WAV file
samples, sample_rate = read_wav_file(wav_file_path)

# Preprocess the audio
samples = preprocess_audio(samples, expected_length)

# Infer using ONNX model
emotion_vector = infer_emotion(onnx_model_path, samples)

print("Emotion Vector:", emotion_vector)
