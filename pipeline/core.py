"""OCR correction signature, program factory and text normalization."""

import random

import dspy
import numpy as np

from .config import SEED

random.seed(SEED)
np.random.seed(SEED)


class OCRCorrection(dspy.Signature):
    """You are given a scene text recognition (OCR) result that may contain errors.
    Correct any recognition errors and output the corrected text.
    If the text appears correct, output it unchanged.
    Only output the corrected text, nothing else."""

    ocr_prediction: str = dspy.InputField(desc="OCR model output that may contain errors")
    corrected_text: str = dspy.OutputField(desc="corrected text")


class PredictCorrector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.correct = dspy.Predict(OCRCorrection)

    def forward(self, ocr_prediction):
        result = self.correct(ocr_prediction=ocr_prediction)
        return dspy.Prediction(corrected_text=result.corrected_text)


def fresh_program(instruction=None):
    program = PredictCorrector()
    if instruction is not None:
        program.correct.signature = OCRCorrection.with_instructions(instruction)
    return program


CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"


def filter_text(text):
    if text is None:
        return ""
    return "".join(c for c in str(text).lower() if c in CHARSET)
