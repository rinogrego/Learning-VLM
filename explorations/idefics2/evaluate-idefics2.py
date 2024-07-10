# !pip install -q git+https://github.com/huggingface/transformers.git
# !pip install -q accelerate datasets peft bitsandbytes
# !pip install tqdm wandb
# !pip install git+https://github.com/huggingface/trl.git

import requests
import random

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from peft import PeftModel, PeftConfig
from transformers.image_utils import load_image
from datasets import load_dataset

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            question = example["question"]
            answer = example['answer']
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels

        return batch
    
def evaluate(model, dataset, batch_size=2, message="test-after-training"):
    import datetime
    stime = datetime.datetime.now()
    model.eval()
    questions = []
    answers = []
    predictions = []
    from tqdm import tqdm
    for idx in tqdm(range(1, len(dataset), batch_size)):
        examples = dataset[idx:idx+batch_size] # return dict
        texts = []
        images = []
        for query, image in zip(examples["question"], examples["image"]):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())
            images.append([image])

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)

        # for q, a, p in zip(examples['question'], examples['answer'], generated_texts):
        #     print("Question\t:", q)
        #     print("Answer\t\t:", a)
        #     print("Predicted\t:", p)

        questions.extend(examples['question'])
        answers.extend(examples["answer"])
        predictions.extend(generated_texts)
        # example["image"].resize((300, 300))
        # plt.show()
    
    import pandas as pd
    df_report = pd.DataFrame({
        "questions": questions,
        "answers": answers,
        "predictions": predictions
    })
    print(df_report.shape)
    df_report.to_excel(f"report-vqa-path/df2_{message}_report.xlsx", index=False)
    etime = datetime.datetime.now()
    total_secs = (etime - stime).total_seconds()
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    seconds = total_secs % 60
    print("Evaluation Time: {} hours | {} minutes | {:.4f} seconds".format(hours, minutes, seconds))
    return df_report

import os
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "VLM-Fine-tuning"

if __name__ == "__main__":
    DATASET = "flaviagiammarino/vqa-rad"
    DEVICE = "cuda:0"
    USE_QLORA = True
    USE_LORA = False
    ADAPTER_MODEL = "rinogrego/idefics2-8b-vqa_path-finetuned"
    
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=False
    )
    
    data_collator = MyDataCollator(processor)
    train_dataset = load_dataset(DATASET, split="train")
    test_dataset = load_dataset(DATASET, split="test")

    config = PeftConfig.from_pretrained(ADAPTER_MODEL)
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = Idefics2ForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config if USE_QLORA else None,
            low_cpu_mem_usage=True
        )
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
        ).to(DEVICE)
    
    # Evaluate pre-training
    print("Evaluate Before Training")
    evaluate(model, train_dataset, message="train-before-training", batch_size=4)
    evaluate(model, test_dataset, message="test-before-training", batch_size=4)
    print("Evaluation finished!")

    model = PeftModel.from_pretrained(model, ADAPTER_MODEL)
    model = model.merge_and_unload()

    # Evaluation
    print("Evaluate After Training")
    evaluate(model, train_dataset, message="train-after-training", batch_size=4)
    evaluate(model, test_dataset, message="test-after-training", batch_size=4)
    print("Evaluation finished!")

