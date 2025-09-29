# train_multimodal.py
import os
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import transformers
from transformers import (
    AutoTokenizer,
    CLIPProcessor,
    CLIPModel,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from llava.model import LlavaLlamaForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 如果 llava.constants 可用可以用它的常量，否则用下边默认
try:
    from llava.constants import IGNORE_INDEX as LLAVA_IGNORE_INDEX, IMAGE_TOKEN_INDEX

    IGNORE_INDEX = LLAVA_IGNORE_INDEX
except Exception:
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = None

DEFAULT_IMAGE_TOKEN = "<image>"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/root/autodl-tmp/model/llava-v1.5-7b")
    vision_tower: str = field(
        default="/root/autodl-tmp/model/clip-vit-large-patch14-336"
    )
    mm_projector_type: str = field(default="linear")


@dataclass
class DataArguments:
    hf_dataset_path: str = field(default="/root/autodl-tmp/data/COCO-Caption2017")
    split: str = field(default="val")


@dataclass
class LoraArguments(TrainingArguments):
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = field(default="q_proj,v_proj")
    output_dir: str = "/root/autodl-tmp/trained"


class COCOCaptionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_processor, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        # ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        question = item["question"] + DEFAULT_IMAGE_TOKEN
        answer = item["answer"][0]
    
        prompt = f"Question: {question}\nAnswer: {answer}"
        input_text = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

        # llava 的 tokenizer_image_token 会把 <image> 占位符转成对应 token id(s)
        from llava.mm_utils import tokenizer_image_token  # import locally

        input_ids = tokenizer_image_token(
            input_text, self.tokenizer, return_tensors="pt"
        ).squeeze(0)

        q_len = len(self.tokenizer(f"Question: {question}\nAnswer: ")["input_ids"])
        labels = input_ids.clone()
        labels[:q_len] = IGNORE_INDEX

        # 图片 -> pixel_values
        pixel_values = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        dict = {
            "input_ids": input_ids[: self.max_length],
            "labels": labels[: self.max_length],
            "pixel_values": pixel_values,
        }
        return dict


def multimodal_collate_fn(batch, tokenizer):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    pixel_values = [b["pixel_values"] for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    pixel_values = torch.stack(pixel_values, dim=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
    }


def find_all_linear_names(model):
    cls = torch.nn.Linear
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, cls) and ("q_proj" in name or "v_proj" in name)
    ]


class MultimodalTrainer(Trainer):
    """
    关键在 compute_loss: 把 vision features 投影到 token embedding 维度，
    然后替换 inputs_embeds 中对应的 <image> token embedding，再 forward LM。
    """

    def __init__(
        self, *args, tokenizer=None, clip_model=None, mm_projector=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.mm_projector = mm_projector

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.device
        # move tensors to model device
        input_ids = inputs["input_ids"].to(device)
        labels = inputs["labels"].to(device)
        pixel_values = inputs["pixel_values"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 1) get image features from CLIP
        with torch.no_grad():
            image_feats = self.clip_model.get_image_features(pixel_values)

        # 2) project to token embedding dim
        projected = self.mm_projector(image_feats)  # (B, token_emb_dim)

        # 3) get token embeddings for input_ids
        inputs_embeds = model.get_input_embeddings()(input_ids)

        # 4) find image token positions
        vocab_size = self.tokenizer.vocab_size
        image_token_id = (
            IMAGE_TOKEN_INDEX
            if IMAGE_TOKEN_INDEX is not None
            else self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        )
        # TODO:修改 token 位置逻辑
        if image_token_id is None or image_token_id >= vocab_size:
            # fallback to unk or first token
            image_token_id = self.tokenizer.unk_token_id

        batch_size, seq_len = input_ids.shape
        for i in range(batch_size):
            ids = input_ids[i]
            # boolean mask for image token positions, clamp idx to seq_len-1
            pos = (ids == image_token_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                idx_pos = 0
            else:
                idx_pos = int(pos[0].item())
                if idx_pos >= seq_len:
                    idx_pos = seq_len - 1

            # replace embedding at idx_pos
            emb = projected[i]  # (emb_dim,)
            inputs_embeds[i, idx_pos, :] = emb

        # 5) forward LM using inputs_embeds
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer + LM
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False
    )
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.float16, device_map="auto"
    )

    # CLIP
    clip_model = CLIPModel.from_pretrained(model_args.vision_tower).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_args.vision_tower)

    # mm_projector: clip_feat_dim -> token_emb_dim
    clip_feat_dim = (
        clip_model.config.projection_dim
        if hasattr(clip_model.config, "projection_dim")
        else clip_model.config.hidden_size
    )
    token_emb_dim = model.get_input_embeddings().weight.shape[1]
    mm_projector = nn.Linear(clip_feat_dim, token_emb_dim).to(model.device)

    # LoRA
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # load hf dataset
    hf_ds = load_dataset(data_args.hf_dataset_path, split=data_args.split)
    dataset = COCOCaptionDataset(hf_ds, tokenizer, clip_processor, max_length=256)

    # print(dataset[0].keys())
    # dict_keys(['input_ids', 'labels', 'pixel_values'])
    # dataLoader = DataLoader(
    #     dataset=dataset,
    #     batch_size=8,
    #     collate_fn=lambda x: multimodal_collate_fn(x, tokenizer),
    # )
    # loader_iter = iter(dataLoader)
    # batch = next(loader_iter)
    # print(batch.keys())
    # dict_keys(['input_ids', 'labels', 'pixel_values', 'attention_mask'])
    # return

    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda x: multimodal_collate_fn(x, tokenizer),
        tokenizer=tokenizer,
        clip_model=clip_model,
        mm_projector=mm_projector,
    )

    trainer.train()

    model.save_pretrained(os.path.join(training_args.output_dir, "lora_weights"))
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
