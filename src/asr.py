import os
import logging

import torch
import librosa
import pytorch_lightning as pl
from nemo.utils import model_utils
from nemo.collections.asr.models import ASRModel

model_path = "/workspace/models/th/best.nemo"


class ASRService:
    def __init__(self):
        ### CPU/GPU Configurations
        if torch.cuda.is_available():
            DEVICE = [0]  # use 0th CUDA device
            ACCELERATOR = "gpu"
        else:
            DEVICE = 1
            ACCELERATOR = "cpu"

        MAP_LOCATION: str = torch.device(
            "cuda:{}".format(DEVICE[0]) if ACCELERATOR == "gpu" else "cpu"
        )

        model_cfg = ASRModel.restore_from(restore_path=model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")

        asr_model = imported_class.restore_from(
            restore_path=model_path,
            map_location=MAP_LOCATION,
        )

        trainer = pl.Trainer(devices=DEVICE, accelerator=ACCELERATOR)
        asr_model.set_trainer(trainer)
        asr_model = asr_model.eval()

        self.asr_model = asr_model

    def transcribe(self, audio_filepath):

        with torch.no_grad():
            transcription = self.asr_model.transcribe(
                paths2audio_files=[
                    audio_filepath,
                ],
                batch_size=1,
                num_workers=0,
                return_hypotheses=False,
            )
        return transcription
