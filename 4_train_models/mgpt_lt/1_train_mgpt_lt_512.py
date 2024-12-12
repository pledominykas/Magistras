from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
import os

transformers.logging.set_verbosity_info()

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

checkpoint_path = "./checkpoints-mgpt-lt-512/checkpoint-12326"

# config = AutoConfig.from_pretrained("ai-forever/mGPT")
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained("domce20/mGPT-lithuanian-tokenizer")

train_dataset = load_from_disk("./datasets/c4-lt-filtered-6-perplexity-100")
eval_dataset = load_from_disk("./datasets/c4-lt-filtered-6-perplexity-validation")

training_args = TrainingArguments(
        resume_from_checkpoint=checkpoint_path,

        optim = "adamw_hf",
        fp16 = True,
        gradient_checkpointing = True,

        num_train_epochs = 1,
        learning_rate = 2e-4,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 16,

        seed = 99,
        output_dir = "./checkpoints-mgpt-lt-512",

        save_strategy = "steps",
        eval_strategy = "steps",

        save_steps = 0.1,
        eval_steps = 0.01,
        logging_steps = 0.01,
        # load_best_model_at_end = True,
    )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,

    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",

    max_seq_length = 512,

    args = training_args,
)

if __name__ == "__main__":
    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.save_model("./models/mgpt-lt-512")
