"""OpenOCR SVTRv2 inference with the true LMDB ID retained through RatioSampler."""

from __future__ import annotations

import argparse
import copy
import io
import sys
import time
import unicodedata
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from prepare import DATASET_PATHS, file_hash, prediction_row, read_selection, save_predictions


class SelectedRatioDataset(Dataset):
    def __init__(self, data_dir, samples, cfg, create_operators, transform):
        self.transform = transform
        self.ops = create_operators(cfg["Eval"]["dataset"]["transforms"], cfg["Global"])
        self.ds_width = cfg["Eval"]["dataset"].get("ds_width", True)
        self.seed = 1
        self.base_shape = cfg["Eval"]["dataset"].get("base_shape", [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.base_h = cfg["Eval"]["dataset"].get("base_h", 32)
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
        self.env = lmdb.open(str(data_dir), readonly=True, lock=False, readahead=False, meminit=False)
        self.items = []
        self.wh_ratio = []
        max_ratio = cfg["Eval"]["loader"].get("max_ratio", 10)
        min_ratio = cfg["Eval"]["loader"].get("min_ratio", 1)
        try:
            with self.env.begin() as txn:
                for sample in samples:
                    idx = sample["image_idx"] + 1
                    label = txn.get(f"label-{idx:09d}".encode())
                    image = txn.get(f"image-{idx:09d}".encode())
                    if label is None or image is None or label.decode("utf-8") != sample["gt"]:
                        raise ValueError(f"missing/changed LMDB record: {sample['dataset']}/{sample['image_idx']}")
                    normalized = unicodedata.normalize("NFKD", "".join(sample["gt"].split())).encode("ascii", "ignore").decode()
                    if len(normalized) > cfg["Global"]["max_text_length"]:
                        raise ValueError(f"SVTRv2 label exceeds supported length: {sample['dataset']}/{sample['image_idx']}")
                    wh = txn.get(f"wh-{idx:09d}".encode())
                    if wh is not None:
                        width, height = map(int, wh.decode().split("_"))
                    else:
                        with Image.open(io.BytesIO(image)) as img:
                            width, height = img.size
                    self.items.append(sample)
                    self.wh_ratio.append(max(min_ratio, min(max_ratio, round(width / height))))
            self.wh_ratio = np.array(self.wh_ratio, dtype=np.int64)
            self.wh_ratio_sort = np.argsort(self.wh_ratio)
        except BaseException:
            self.env.close()
            raise

    def __len__(self):
        return len(self.items)

    def __getitem__(self, properties):
        sample = self.items[int(properties[2])]
        ratio = int(properties[3])
        with self.env.begin() as txn:
            image = txn.get(f"image-{sample['image_idx'] + 1:09d}".encode())
        data = self.transform({"image": image, "label": sample["gt"]}, self.ops[:-1])
        if data is None:
            raise ValueError(f"SVTRv2 transform failed: {sample['dataset']}/{sample['image_idx']}")
        img = data["image"]
        width, height = img.size
        img_w, img_h = self.base_shape[ratio - 1] if ratio <= 4 else [self.base_h * ratio, self.base_h]
        if img_w // img_h >= width // height + 2:
            raise ValueError(f"SVTRv2 resize failed: {sample['dataset']}/{sample['image_idx']}")
        data["image"] = self.transforms(F.resize(img, (img_h, img_w), interpolation=T.InterpolationMode.BICUBIC))
        data["valid_ratio"] = 1.0
        data["gen_ratio"] = img_w // img_h
        data["real_ratio"] = max(1, round(width / height))
        output = self.transform(data, self.ops[-1:])
        if output is None:
            raise ValueError(f"SVTRv2 KeepKeys failed: {sample['dataset']}/{sample['image_idx']}")
        return output

    def close(self):
        self.env.close()


def resolve_dict_paths(value, root):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "character_dict_path" and item:
                value[key] = str((root / item).resolve())
            else:
                resolve_dict_paths(item, root)
    elif isinstance(value, list):
        for item in value:
            resolve_dict_paths(item, root)


def evaluate_model(args):
    started = time.monotonic()
    root = args.openocr_root.resolve()
    if not (root / "openrec").is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("OpenOCR checkout or SVTRv2 checkpoint is missing")
    sys.path.insert(0, str(root))
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from openrec.preprocess import create_operators, transform
    from tools.data.ratio_sampler import RatioSampler
    from tools.engine.config import Config

    selected = read_selection(args.output_dir / "selection.csv")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; explicitly select --device cpu")
    cfg = Config(str(args.config)).cfg
    resolve_dict_paths(cfg, root)
    cfg["PostProcess"]["with_ratio"] = True
    keep_keys = cfg["Eval"]["dataset"]["transforms"][-1]["KeepKeys"]["keep_keys"]
    if "real_ratio" not in keep_keys:
        keep_keys.append("real_ratio")
    postprocess = build_post_process(cfg["PostProcess"], cfg["Global"])
    cfg["Architecture"]["Decoder"]["out_channels"] = postprocess.get_character_num()
    model = build_model(cfg["Architecture"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    decoder = cfg["Architecture"]["Decoder"]
    # The official CTC-only inference config does not construct the training GTC branch.
    if decoder["name"] == "GTCDecoder" and not decoder.get("infer_gtc", False):
        state = {key: value for key, value in state.items() if not key.startswith("decoder.gtc_decoder.")}
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    rows = []
    for name in DATASET_PATHS:
        samples = [row for row in selected if row["dataset"] == name]
        if not samples:
            continue
        dataset = SelectedRatioDataset(args.data_root / DATASET_PATHS[name], samples, cfg, create_operators, transform)
        try:
            sampler_cfg = copy.deepcopy(cfg["Eval"]["sampler"])
            sampler_cfg.pop("name", None)
            sampler_cfg.update(first_bs=args.batch_size, is_training=False)
            sampler = RatioSampler(dataset, **sampler_cfg)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)
            with torch.no_grad():
                for batch_index, batch in enumerate(loader):
                    items = [dataset.items[int(prop[2])] for prop in sampler.batch_list[batch_index]]
                    device_batch = [item.to(device) if hasattr(item, "to") else item for item in batch]
                    numpy_batch = [item.numpy() if hasattr(item, "numpy") else item for item in batch]
                    preds = model(device_batch[0], data=device_batch[1:])
                    predictions, _ = postprocess(preds, numpy_batch)
                    if len(predictions) != len(items):
                        raise RuntimeError("SVTRv2 batch size mismatch")
                    rows.extend(prediction_row("svtrv2", sample, pred[0]) for sample, pred in zip(items, predictions))
        finally:
            dataset.close()
    provenance = {"framework": "Topdu/OpenOCR", "config_sha256": file_hash(args.config),
                  "checkpoint_sha256": file_hash(args.checkpoint), "device": str(device),
                  "batch_size": args.batch_size, "torch_version": torch.__version__,
                  "amp": False, "sampler": "RatioSampler with original LMDB IDs"}
    return save_predictions(args.output_dir, "svtrv2", rows, selected, provenance, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    evaluate_model(args)


if __name__ == "__main__":
    main()
