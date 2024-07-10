# !pip install -q git+https://github.com/huggingface/transformers.git
# !pip install -q accelerate datasets peft bitsandbytes
# !pip install tqdm wandb
# !pip install git+https://github.com/huggingface/trl.git

import requests
import random

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from peft import LoraConfig
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
    df_report.to_excel(f"report/df_{message}_report.xlsx", index=False)
    etime = datetime.datetime.now()
    total_secs = (etime - stime).total_seconds()
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    seconds = total_secs % 60
    print("Evaluation Time: {} hours | {} minutes | {} seconds".format(hours, minutes, seconds))
    return df_report

import os
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "VLM-Fine-tuning"

if __name__ == "__main__":
    DATASET = "vqa-rad"
    # pmc_vqa = load_dataset("xmcmic/PMC-VQA")
    # path_vqa = load_dataset("flaviagiammarino/path-vqa")
    # vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
    
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=False
    )
    
    data_collator = MyDataCollator(processor)
    # train_dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")
    # test_dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
    train_dataset = load_dataset("flaviagiammarino/path-vqa", split="train")
    test_dataset = load_dataset("flaviagiammarino/path-vqa", split="test")

    DEVICE = "cuda:0"
    USE_QLORA = True
    USE_LORA = False
    # Three options for training, from the lowest precision training to the highest precision training:
    # - QLora
    # - Standard Lora
    # - Full fine-tuning
    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian"
        )
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            quantization_config=bnb_config if USE_QLORA else None,
            low_cpu_mem_usage=True
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
        ).to(DEVICE)
    
    # Evaluate pre-training
    # print("Evaluate Before Training")
    # # evaluate(model, train_dataset, message="train-before-training", batch_size=4)
    # evaluate(model, test_dataset, message="test-before-training", batch_size=4)
    # print("Evaluation finished!")

    training_args = TrainingArguments(
        num_train_epochs = 3,
        # max_steps = 2,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = 1,
        # warmup_steps = 50,
        warmup_ratio = 0.3,
        learning_rate = 3e-4,
        weight_decay = 0.2,
        logging_steps = 100, # record train loss per mentioned step
        output_dir = "idefics2/",
        save_strategy = "steps",
        save_steps = 5000,
        eval_steps = 2000, # record validation loss per mentioned step
        save_total_limit = 1,
        eval_strategy = "steps",
        fp16 = True,
        hub_model_id = "idefics2-8b-vqa_path-finetuned",
        remove_unused_columns = False,
        report_to = "wandb",
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = test_dataset
    )
    
    trainer.train()
    
    trainer.push_to_hub(token=HF_TOKEN)

    ## Evaluation
    # print("Evaluate After Training")
    # # evaluate(model, train_dataset, message="train-after-training", batch_size=4)
    # evaluate(model, test_dataset, message="test-after-training", batch_size=4)
    # print("Evaluation finished!")

