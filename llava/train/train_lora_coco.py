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
    from llava.constants import IGNORE_INDEX as LLAVA_IGNORE_INDEX
    from llava.constants import IMAGE_TOKEN_INDEX

    IGNORE_INDEX = LLAVA_IGNORE_INDEX
except Exception:
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200

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
        question = item["question"]
        answer = item["answer"][0]

        prompt = f"Question: {question}\nAnswer: {answer}"

        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(
            0
        )  # (L,)
        image_token_id = self.tokenizer.pad_token_id
        input_ids = torch.cat(
            [torch.tensor([image_token_id], dtype=torch.long), input_ids], dim=0
        )

        q_len = len(self.tokenizer(f"Question: {question}\nAnswer: ")["input_ids"])
        labels = input_ids.clone()
        labels[1 : q_len + 1] = IGNORE_INDEX

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
    Multimodal Trainer:
    - 把 CLIP 提取的 vision features 投影到 LLM 的 embedding 空间
    - 替换掉 input_ids 里对应 <image> token 的 embedding
    """

    def __init__(
        self,
        *args,
        tokenizer=None,
        clip_model=None,
        mm_projector=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.mm_projector = mm_projector

        # 确保 tokenizer 里有 <image> token
        # if image_token not in tokenizer.get_vocab():
        #     tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
        #     print(f"✅ Added {image_token} to tokenizer")
        # self.image_token = image_token
        # self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

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
        # === 1. Move tensors to device ===
        input_ids = inputs["input_ids"].to(device)
        labels = inputs["labels"].to(device)
        pixel_values = inputs["pixel_values"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        # === 2. Vision Encoder (CLIP) ===
        with torch.no_grad():
            image_feats = self.clip_model.get_image_features(
                pixel_values
            )  # (B, D_clip)
        # === 3. Project to LM embedding dim ===
        projected = self.mm_projector(image_feats)  # (B, D_lm)
        # === 4. Replace <image> token embeddings ===
        inputs_embeds = model.get_input_embeddings()(input_ids)  # (B, L, D_lm)

        batch_size, seq_len = input_ids.shape
        for i in range(batch_size):
            idx_pos = 0
            inputs_embeds[i, idx_pos, :] = projected[i]

        # === 5. Forward LM ===
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
    model.resize_token_embeddings(len(tokenizer))

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
    torch.save(
        mm_projector.state_dict(),
        os.path.join(training_args.output_dir, "mm_projector.pt"),
    )
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
