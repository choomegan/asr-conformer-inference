import os
from typing import List, Union, Dict, Optional

import torch
import numpy as np
from nemo.collections.asr.modules import BeamSearchDecoderWithLM


def softmax(x):
    """
    numpy implementation of softmax
    """
    x = x.cpu().numpy()
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])


class LMInference:
    """
    ASR Inference class
    """

    def __init__(
        self,
        lang: str,
        model_path: str,
        vocab: List[Union[str, int]],
        ids_to_text_func: Optional[Dict[str, str]] = None,
        token_offset: int = 100,
    ) -> None:

        if lang == "th":
            self.beam_width = 32
            self.alpha = 0.75
            self.beta = 1
        elif lang == "vi" or lang == "tl":
            self.beam_width = 128
            self.alpha = 1
            self.beta = 1
        elif lang == "bn":
            self.beam_width = 64
            self.alpha = 1
            self.beta = 1
        elif lang == "en":
            self.beam_width = 16
            self.alpha = 2
            self.beta = 1.5
        else:
            raise Exception(
                "Language not supported! Only vi, th and tl and bn (and en)"
            )

        self.vocab = vocab
        self.ids_to_text_func = ids_to_text_func
        self.token_offset = token_offset

        self.lm = BeamSearchDecoderWithLM(
            vocab=vocab,
            beam_width=self.beam_width,
            alpha=self.alpha,
            beta=self.beta,
            lm_path=model_path,
            num_cpus=max(os.cpu_count(), 1),
            input_tensor=False,
        )

    def transcribe(self, logprobs: torch.tensor) -> str:
        """
        Transcribe function
        """
        if logprobs.is_cuda:
            logprobs = logprobs.cpu()

        softmaxed_logprobs = softmax(logprobs)
        beams = self.lm.forward(
            log_probs=np.expand_dims(softmaxed_logprobs, axis=0),
            log_probs_length=None,
        )[0]

        # only use the best candidate
        best_candidate = beams[0]

        if self.ids_to_text_func is not None:
            # For BPE encodings, need to shift by TOKEN_OFFSET to retrieve the original sub-word ids
            hypothesis = self.ids_to_text_func(
                [ord(c) - self.token_offset for c in best_candidate[1]]
            )
        else:
            hypothesis = best_candidate[1]

        return hypothesis
