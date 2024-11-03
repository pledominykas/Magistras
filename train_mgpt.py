from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from transformers import TrainingArguments
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

n_gpus = torch.cuda.device_count()

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")

if n_gpus > 1:
    model = DDP(model, device_ids = [i for i in range(n_gpus)])

train_dataset = load_from_disk("./output-k-200-cleaned-no-bad-words")
eval_dataset = load_from_disk("./c4-lithuanian-validation")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,

    args = TrainingArguments(                             
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,
        
        num_train_epochs = 3,
        learning_rate = 2e-4,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,

        seed = 99,
        output_dir = "./checkpoints-mgpt",

        save_strategy = "steps",
        eval_strategy = "steps",

        save_steps = 0.1,
        eval_steps = 0.1,
        logging_steps = 0.1
    ),
)

trainer_stats = trainer.train()

with open("output-mgpt-train.json", "w") as file:
    file.write(json.dumps(trainer.state.log_history))