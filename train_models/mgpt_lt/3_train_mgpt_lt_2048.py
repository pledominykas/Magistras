from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
import os

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

train_dataset = load_from_disk("./datasets/c4-lt-filtered-6-perplexity-1")
eval_dataset = load_from_disk("./datasets/c4-lt-filtered-6-perplexity-validation")

model = AutoModelForCausalLM.from_pretrained("./models/mgpt-lt-1024")
tokenizer = AutoTokenizer.from_pretrained("domce20/mGPT-lithuanian-tokenizer")

training_args = TrainingArguments(
        optim = "adamw_bnb_8bit",
        fp16 = True,
        gradient_checkpointing = True,

        num_train_epochs = 1,
        learning_rate = 2e-4,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = 8,

        seed = 99,
        output_dir = "./checkpoints-mgpt-lt-2048",

        save_strategy = "steps",
        eval_strategy = "steps",

        save_steps = 0.5,
        eval_steps = 0.1,
        logging_steps = 0.1,
        # load_best_model_at_end = True,
    )



trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,

    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",

    max_seq_length = 2048,

    args = training_args,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./models/mgpt-lt-2048")
