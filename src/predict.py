"""
Main script to do inference on audio files in a specified dir
"""

import os

import tqdm
import json
import torch
import librosa

from config import config
from inference_models.asr import ASRInference
from inference_models.lm import LMInference

""" CPU/GPU Configurations """
if torch.cuda.is_available():
    DEVICE = [0]  # use 0th CUDA device
    ACCELERATOR = "gpu"
else:
    DEVICE = 1
    ACCELERATOR = "cpu"

""" Initialize Models """
SAMPLE_RATE = 16000

asr_model = ASRInference(config.asr_model_path, DEVICE, ACCELERATOR)

vocab = [chr(idx + config.token_offset) for idx in range(len(asr_model.vocabulary))]
lm = LMInference(
    lang=config.language,
    model_path=config.lm_path,
    vocab=vocab,
    ids_to_text_func=asr_model.ids_to_text_func,
    token_offset=config.token_offset,
)


def predict(audio_path: str) -> str:
    """
    run through entire inference pipeline
    """

    # loads and resamples to input sample_rate and convert to mono/stereo
    audio, _ = librosa.load(path=audio_path, sr=SAMPLE_RATE, mono=True)

    audio_tensor = torch.tensor(audio)

    audio_length_tensor = torch.tensor(audio_tensor.shape)
    audio_tensor = audio_tensor.unsqueeze(0)  # add new dim at index 0

    if config.use_lm:
        logprobs = asr_model.transcribe(
            audio_tensor, audio_length_tensor, return_logprobs=True
        )
        transcription = lm.transcribe(logprobs)

    else:  # inference without LM
        transcription = asr_model.transcribe(
            audio_tensor, audio_length_tensor, return_logprobs=False
        )

    return transcription


def main():
    if config.audio_dir:
        print("Transcrbing an audio folder........")

        filenames = os.listdir(config.audio_dir)
        filepaths = [os.path.join(config.audio_dir, fn) for fn in filenames]

        for _, audio_file in tqdm.tqdm(enumerate(filepaths), total=len(filepaths)):
            with open(config.output_manifest_path, "a", encoding="utf-8") as fw:
                duration = librosa.get_duration(path=audio_file, sr=16000)
                transcript = predict(audio_file)

                metadata = {
                    "audio_filepath": audio_file,
                    "duration": duration,
                    "pred_text": transcript,
                }

                fw.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    else:
        # transcribe through manifest file
        print("Transcrbing via a manifest file........")
        items = []
        with open(config.input_manifest_path, "r+", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))

        for item in tqdm.tqdm(items):
            with open(config.output_manifest_path, "a", encoding="utf-8") as fw:
                audio_file = os.path.join(
                    os.path.dirname(config.input_manifest_path),
                    item["audio_filepath"],
                )
                duration = librosa.get_duration(path=audio_file, sr=16000)
                transcript = predict(audio_file)

                item["pred_text"] = transcript

                fw.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
