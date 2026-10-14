"""Run one PARSeq-distributed recognizer on the already selected LMDB images."""

import argparse
import io
import time
from pathlib import Path
from urllib.parse import urlparse

import lmdb
import torch
from PIL import Image
from torchvision import transforms as T

from prepare import DATASET_PATHS, MODELS, file_hash, prediction_row, read_selection, save_predictions


def load_model(model_name, device, parseq_root=None):
    if parseq_root:
        model = torch.hub.load(str(parseq_root), model_name, source="local", pretrained=True)
    else:
        model = torch.hub.load("baudm/parseq", model_name, pretrained=True, trust_repo=True)
    return model.eval().to(device)


def get_transform(model):
    return T.Compose([
        T.Resize(model.hparams.get("img_size", [32, 128]), T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(0.5, 0.5),
    ])


def evaluate_model(args):
    started = time.monotonic()
    selected = read_selection(args.output_dir / "selection.csv")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; explicitly select --device cpu")
    model = load_model(args.model, device, args.parseq_root)
    img_transform = get_transform(model)
    rows = []
    for dataset in DATASET_PATHS:
        samples = [row for row in selected if row["dataset"] == dataset]
        if not samples:
            continue
        env = lmdb.open(str(args.data_root / DATASET_PATHS[dataset]), readonly=True, lock=False, readahead=False)
        try:
            with env.begin() as txn, torch.no_grad():
                for start in range(0, len(samples), args.batch_size):
                    batch = samples[start:start + args.batch_size]
                    images = []
                    for sample in batch:
                        idx = sample["image_idx"] + 1
                        image = txn.get(f"image-{idx:09d}".encode())
                        label = txn.get(f"label-{idx:09d}".encode())
                        if image is None or label is None or label.decode("utf-8") != sample["gt"]:
                            raise ValueError(f"missing/changed LMDB record: {dataset}/{sample['image_idx']}")
                        with Image.open(io.BytesIO(image)) as img:
                            images.append(img_transform(img.convert("RGB")))
                    logits = model(torch.stack(images).to(device))
                    predictions = model.tokenizer.decode(logits)[0]
                    if len(predictions) != len(batch):
                        raise RuntimeError("OCR batch size mismatch")
                    rows.extend(prediction_row(args.model, sample, pred) for sample, pred in zip(batch, predictions))
        finally:
            env.close()
    from strhub.models.utils import _WEIGHTS_URL

    url = _WEIGHTS_URL[args.model]
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name
    provenance = {"framework": "baudm/parseq", "weights_url": url,
                  "checkpoint_sha256": file_hash(checkpoint), "device": str(device),
                  "batch_size": args.batch_size, "torch_version": torch.__version__}
    return save_predictions(args.output_dir, args.model, rows, selected, provenance, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS[:-1])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parseq-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    evaluate_model(args)


if __name__ == "__main__":
    main()
