import hashlib
import os
import urllib
import warnings
import zipfile
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer


import glob






try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

if torch.__version__.split(".") < ["1", "7", "1"]:
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "CLIP_ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "RemoteCLIP_ViT-B/32": "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt",
    "GeoRSCLIP_ViT-B/32": "https://huggingface.co/Zilun/GeoRSCLIP/resolve/main/ckpt/RS5M_ViT-B-32.pt",
    "SkyCLIP_ViT-B/32": "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip",
    "CLIP_ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",    
    "RemoteCLIP_ViT-L/14": "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-L-14.pt",
    "GeoRSCLIP_ViT-L/14": "https://huggingface.co/Zilun/GeoRSCLIP/resolve/main/ckpt/RS5M_ViT-L-14.pt",
    "SkyCLIP_ViT-L/14": "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_L14_top50pct.zip",

    "GeoRSCLIP_ViT-H/14": "https://huggingface.co/Zilun/GeoRSCLIP/resolve/main/ckpt/RS5M_ViT-H-14.pt",

}



def _download(url: str, root: str = os.path.expanduser("~/.cache/clip"), expected_sha256: str = None, extract: bool = False):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)

    if os.path.isfile(download_target):
        if expected_sha256:
            file_checksum = hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            if file_checksum == expected_sha256:
                print(f"File already exists and checksum matches: {download_target}")
            else:
                warnings.warn(
                    f"File exists, but checksum does not match: {download_target}\n"
                    f"Expected: {expected_sha256}, Found: {file_checksum}. Re-downloading..."
                )
        else:
            print(f"File already exists, skipping download: {download_target}")
            if extract and zipfile.is_zipfile(download_target):
                extraction_dir = os.path.join(root, filename.replace('.zip', ''))
                if os.path.isdir(extraction_dir):
                    print(f"Extraction directory already exists: {extraction_dir}")
                    return extraction_dir
            return download_target

    print(f"Downloading {url} to {download_target}")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length", 0)), 
            ncols=80, 
            unit="iB", 
            unit_scale=True
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256:
        file_checksum = hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        if file_checksum != expected_sha256:
            raise RuntimeError(
                f"Model has been downloaded but the SHA256 checksum does not match.\n"
                f"Expected: {expected_sha256}, Found: {file_checksum}"
            )
        print(f"Checksum verified: {file_checksum}")

    if extract:
        if zipfile.is_zipfile(download_target):
            extraction_dir = os.path.join(root, filename.replace('.zip', ''))
            if not os.path.isdir(extraction_dir):
                print(f"Extracting {download_target} to {extraction_dir}")
                with zipfile.ZipFile(download_target, 'r') as zip_ref:
                    zip_ref.extractall(extraction_dir)
            else:
                print(f"Extraction directory already exists: {extraction_dir}")
            return extraction_dir
        else:
            warnings.warn(f"The downloaded file is not a ZIP archive: {download_target}")

    return download_target


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=False):
    if name in _MODELS:
        if name == "SkyCLIP_ViT-B/32":
            model_path = _download(_MODELS[name], extract=True)
            
            # Manually locate the checkpoint file
            checkpoint_path = os.path.join(model_path, "SkyCLIP_ViT_B32_top50pct", "epoch_20.pt")
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"SkyCLIP ViT-B/32 checkpoint not found: {checkpoint_path}")
            
            # Load the state dictionary
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        elif name == "SkyCLIP_ViT-L/14":
            model_path = _download(_MODELS[name], extract=True)
            
            # Manually locate the checkpoint file
            checkpoint_path = os.path.join(model_path, "SkyCLIP_ViT_L14_top50pct", "epoch_20.pt")
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"SkyCLIP ViT-L/14 checkpoint not found: {checkpoint_path}")
            
            # Load the state dictionary
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            model_path = _download(_MODELS[name], extract=name.endswith(".zip"))
            state_dict = None
    elif os.path.isfile(name):
        model_path = name
        state_dict = None
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # Build the model
    model = build_model(state_dict)

    # Load state dictionary into the model
    if state_dict:
        model.load_state_dict(state_dict, strict=False)

    return model, _transform(model.visual.input_resolution)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
