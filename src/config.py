from pydantic import BaseSettings, Field
from typing import Optional


class BaseConfig(BaseSettings):
    """Define any config here.

    See here for documentation:
    https://pydantic-docs.helpmanual.io/usage/settings/
    """

    # language: str = "th"  # vi, th or tl
    # asr_model_path: str = f"/home/app/MALT_models/{language}/best.nemo"

    # use_lm: bool = True  # whether to use LM or not
    # lm_path: str = f"/home/app/MALT_models/{language}/kenlm_model.bin"
    # token_offset: int = 100

    # audio_dir: str = f"/home/app/language_id/batch_2/{language}"
    # output_manifest_path: str = (
    #     f"/home/app/MALT_outputs/{language}/conformer_with_lm.json"
    # )
    language: str = "en"
    asr_model_path: f"/workspace/models/{language}/best.nemo"

    use_lm: bool = False  # whether to use LM or not
    lm_path: str = f"/workspace/models/{language}/kenlm_model.bin"
    token_offset: int = 100

    # audio_dir: Optional[str] = "/datasets/voice-restore/mms_restored/train"
    # output_manifest_path: str = f"/datasets/voice-restore/fastconformer_with_lm.json"

    audio_dir: Optional[str] = None
    input_manifest_path: Optional[str] = (
        "/datasets/PORT/mms_set_1/test_split/test_manifest_sampled_vocal_removed.json"
    )
    output_manifest_path: str = (
        f"/datasets/PORT/mms_set_1/test_split/test_manifest_sampled_orig_base_model.json"
    )


config = BaseConfig()
