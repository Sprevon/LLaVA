"""
LLaVA lora 微调 手撕
"""
import transformers
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from llava.train.llava_trainer import LLaVATrainer

# FIXME
tokenizer_path = ""
processor_path = ""
dataset_path = ""
model_path = ""

class SFTDataset(Dataset):
    """
    加载数据集，目前是 COCO-Caption2017
    """
    def __init__(self, tokenizer, processor):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.data = load_dataset(dataset_path, split="val")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image = item['image']
        text = item['answer'][0]

        # 处理文本
        text_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        # 处理图像
        image_tokens = self.processor(
            images=image,
            return_tensors="pt"
        )

        # 打包成一个 dict
        return {
            "input_ids": text_tokens["input_ids"].squeeze(0),
            "attention_mask": text_tokens["attention_mask"].squeeze(0),
            "pixel_values": image_tokens["pixel_values"].squeeze(0),
        }


# class SFTCollator:
#     def __init__(self, tokenizer, processor, max_length) -> None:
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.max_length = max_length

#     def __call__(self, batch):
#         # TODO: 数据预处理
        



def train():
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    clip_processor = transformers.CLIPProcessor.from_pretrained(processor_path)
    model = transformers.LlavaModel.from_pretrained(model_path)
    dataset = SFTDataset(text_tokenizer, clip_processor)
    # collator_fn = SFTCollator(text_tokenizer, clip_processor)
    # trainer = LLaVATrainer(model=model, train_dataset=dataset,
    #                         eval_dataset=None, data_collator=collator_fn)
    # trainer.train()
