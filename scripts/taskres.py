'''
Task Residual Tuning
by Tao Yu (yutao666@mail.ustc.edu.cn)
Oct 4, 2022
'''
import os
import os.path as osp
from re import template

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "aid": "a satellite photo of {}.",
    "eurosat": "a satellite photo of {}.",
    "mlrsnet": "a satellite photo of {}.",
    "optimal31": "a satellite photo of {}.",
    "patternnet": "a satellite photo of {}.",
    "resisc45": "a satellite photo of {}.",
    "rsc11": "a satellite photo of {}.",
    "rsicb128": "a satellite photo of {}.",
    "rsicb256": "a satellite photo of {}.",
    "whurs19": "a satellite photo of {}.",
}

#
# NEW, IMPROVED SCHEDULER CODE
#
import torch
from torch.optim.lr_scheduler import _LRScheduler

AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        # FIX APPLIED: The 'verbose' argument is removed from the super() call
        # to ensure compatibility with modern PyTorch versions.
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]

# NOTE: You might need to add this import if it's not already in taskres.py
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler."""
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.get("STEPSIZE", [30]) # Use .get() for safety
    gamma = optim_cfg.get("GAMMA", 0.1)
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]
        scheduler = StepLR(optimizer, step_size=stepsize, gamma=gamma)

    elif lr_scheduler == "multi_step":
        scheduler = MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    elif lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, float(max_epoch))

    # Safely check for warmup settings
    if hasattr(optim_cfg, 'WARMUP_EPOCH') and optim_cfg.WARMUP_EPOCH > 0:
        if optim_cfg.WARMUP_TYPE == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_CONS_LR
            )
        elif optim_cfg.WARMUP_TYPE == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_MIN_LR
            )
        else:
            raise ValueError(f"Unsupported warmup type: {optim_cfg.WARMUP_TYPE}")

    return scheduler

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    try:
        if backbone_name.startswith("SkyCLIP"):
            model, preprocess = clip.load(backbone_name, device="cpu")

            base_path = os.path.expanduser("~/.cache/clip")
            if backbone_name == "SkyCLIP_ViT-B/32":
                model_dir = os.path.join(base_path, "SkyCLIP_ViT_B32_top50pct")
            elif backbone_name == "SkyCLIP_ViT-L/14":
                model_dir = os.path.join(base_path, "SkyCLIP_ViT_L14_top50pct")
            else:
                raise RuntimeError(f"Unsupported SkyCLIP model: {backbone_name}")

            # Locate the checkpoint file
            checkpoint_path = None
            for root, _, files in os.walk(model_dir):
                if "epoch_20.pt" in files:
                    checkpoint_path = os.path.join(root, "epoch_20.pt")
                    break

            if not checkpoint_path:
                raise RuntimeError(f"SkyCLIP checkpoint not found in directory: {model_dir}")

            print(f"Checkpoint found at: {checkpoint_path}")

            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print("Checkpoint keys:", list(checkpoint.keys()))  # Debugging checkpoint keys

            # Extract and strip the prefix from 'state_dict'
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
            print(f"State dict type before returning: {type(state_dict)}")
            print("State dict keys (after):", list(state_dict.keys())[:10])

            # Load the state_dict into the model
            model.load_state_dict(state_dict, strict=False)

        else:
            # For non-SkyCLIP models, use clip.load() directly
            model, preprocess = clip.load(backbone_name, device="cpu")

        print(f"Type of model being returned: {type(model)}")
        return model  # Ensure we're returning the `model`, not the `TaskRes` object.

    except RuntimeError as e:
        raise RuntimeError(f"Error loading CLIP model '{backbone_name}': {e}")

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

# TaskRes(-Text)
class TaskResLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, base_text_features):
        super().__init__()
        self.device = clip_model.dtype
        self.alpha = cfg.TRAINER.TaskRes.RESIDUAL_SCALE
        print(">> DCT scale factor: ", self.alpha)
        self.register_buffer("base_text_features", base_text_features)
        self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

    def forward(self):
        return self.base_text_features + self.alpha * self.text_feature_residuals   # t + a * x

# # TaskRes-Image
# class TaskResLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model, base_text_features):
#         super().__init__()
#         self.device = clip_model.dtype
#         # feat_dim = base_text_features.size(-1)
#         self.alpha = cfg.TRAINER.TaskRes.RESIDUAL_SCALE
#         print(">> DCT scale factor: ", self.alpha)
#         self.register_buffer("base_text_features", base_text_features)
#         self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features[0:1]))

#     def forward(self):
#         # print(self.base_text_features.dtype, self.text_feature_residuals.dtype)
#         return self.base_text_features, self.alpha * self.text_feature_residuals

def _get_base_text_features(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()
    
    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)

def _get_enhanced_base_text_features(cfg, classnames, clip_model, text_encoder, pretraiend_model):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

        pretrained_text_projection = torch.load(pretraiend_model)

        state_dict = text_encoder.state_dict()
        state_dict['text_projection'] = pretrained_text_projection['state_dict']['weight'].t()
        text_encoder.load_state_dict(state_dict)
        print(">> Pretrained text encoder loaded!")
        params = pretrained_text_projection['state_dict']['weight'].size(0) * \
            pretrained_text_projection['state_dict']['weight'].size(1)
        print(">> Text projection parameters: ", params)
        print(pretrained_text_projection['state_dict'].keys())
    
    dataset = cfg.DATASET.NAME
    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)

# TaskRes by Tao Yu, Oct 4, 2022
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype   # float16
        text_encoder = TextEncoder(clip_model)
        if cfg.TRAINER.TaskRes.ENHANCED_BASE == "none":
            print(">> Use regular base!")
            base_text_features = _get_base_text_features(cfg, classnames, clip_model, text_encoder)
        else:
            print(">> Use enhanced base!")
            base_text_features = _get_enhanced_base_text_features(
                cfg, classnames, clip_model, text_encoder, cfg.TRAINER.TaskRes.ENHANCED_BASE)

        self.prompt_learner = TaskResLearner(cfg, classnames, clip_model, base_text_features)

    def forward(self, image):
        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except:
            image_features = self.image_encoder(image.float())

        # TaskRes-Text
        text_features = self.prompt_learner()

        # # TaskRes-Image
        # text_features, image_res = self.prompt_learner()
        # image_features += image_res

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

@TRAINER_REGISTRY.register()
class TaskRes(TrainerX):
    """Context Optimization (TaskRes).

    Task Residual for Tuning Vision-Language Models
    https://arxiv.org/abs/2211.10277
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TaskRes.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # Debugging: Check type of clip_model
        print(f"Type of clip_model: {type(clip_model)}")

        if cfg.TRAINER.TaskRes.PREC == "fp32" or cfg.TRAINER.TaskRes.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # Debugging: Check type of `self.model`
        print(f"Type of self.model: {type(self.model)}")

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model = self.model.float()
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.TaskRes.PREC == "amp" else None

        # Note: Multi-GPU training could be slow because CLIP's size is large
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.TaskRes.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            if self.cfg.DATASET.NAME == 'ImageNetA' or self.cfg.DATASET.NAME == 'ImageNetR':
                if self.cfg.DATASET.NAME == 'ImageNetA':
                    from .imagenet_a_r_indexes_v2 import find_imagenet_a_indexes as find_indexes
                else:
                    from .imagenet_a_r_indexes_v2 import find_imagenet_r_indexes as find_indexes
                imageneta_indexes = find_indexes()
                state_dict['base_text_features'] = state_dict['base_text_features'][imageneta_indexes]
                state_dict['text_feature_residuals'] = state_dict['text_feature_residuals'][imageneta_indexes]

            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)