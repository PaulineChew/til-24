import torch
import torchaudio

class ASRManager:
    def __init__(self, model_name="openai/whisper-large-v3"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)

    def transcribe(self, audio_bytes: bytes) -> str:
        waveform, sample_rate = librosa.load(audio_bytes)
        input_features = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features

        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription