from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from datasets import load_dataset

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("utter-project/EuroLLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("utter-project/EuroLLM-1.7B")

tokenizer.pad_token = tokenizer.eos_token

model = model.to(device)

eval_dataset = load_dataset("allenai/c4", "lt", split="validation")
train_dataset = load_from_disk("./output-k-200-cleaned-no-bad-words")

train_dataset = train_dataset.take(1000000)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field="text",
    max_seq_length = 4096,

    eval_dataset = eval_dataset,

    args = TrainingArguments(
        auto_find_batch_size = True,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        seed = 99,
        output_dir = "./checkpoints",

        save_strategy = "steps",
        eval_strategy = "steps",

        save_steps = 0.1, # Create 10 checkpoints during training
        eval_steps = 0.1, # Evaluate 10 times increse to 30 for full training
        logging_steps = 0.1
    ),
)

trainer_stats = trainer.train()