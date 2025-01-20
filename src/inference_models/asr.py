import logging
from typing import List, Union

import torch
import pytorch_lightning as pl
from nemo.utils import model_utils
from nemo.collections.asr.models import ASRModel


class ASRInference:
    """
    ASR Inference class
    """

    def __init__(
        self, model_path: str, device: Union[str, List[int]], accelerator: str
    ) -> None:

        self.map_location = torch.device(
            f"cuda:{device[0]}" if accelerator == "gpu" else "cpu"
        )

        model_cfg = ASRModel.restore_from(restore_path=model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info("Restoring model : %s", imported_class.__name__)

        asr_model = imported_class.restore_from(
            restore_path=model_path,
            map_location=self.map_location,
        )

        trainer = pl.Trainer(devices=device, accelerator=accelerator)
        asr_model.set_trainer(trainer)

        self.asr_model = asr_model.eval()
        self.vocabulary = self.asr_model.decoder.vocabulary
        self.ids_to_text_func = self.asr_model.tokenizer.ids_to_text

    def transcribe(
        self,
        audio_tensor: torch.tensor,
        audio_length_tensor: torch.tensor,
        return_logprobs: bool = False,
    ) -> torch.tensor:
        """
        Transcribe function
        """

        with torch.no_grad():
            logits, logits_len, greedy_predictions = self.asr_model.forward(
                input_signal=audio_tensor.to(self.map_location),
                input_signal_length=audio_length_tensor.to(self.map_location),
            )

        if return_logprobs:
            del logits_len, greedy_predictions
            # return Shape: (T, D)
            return logits.squeeze(dim=0)

        with torch.no_grad():
            hypotheses, all_hyp = (
                self.asr_model.decoding.ctc_decoder_predictions_tensor(
                    logits,
                    decoder_lengths=logits_len,
                )
            )
        del all_hyp
        return hypotheses[0]
