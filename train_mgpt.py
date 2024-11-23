import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from transformers import TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")

train_dataset = load_dataset("domce20/c4-lithuanian-final")
eval_dataset = load_dataset("domce20/c4-lithuanian-validation")

train_dataset = train_dataset["train"]
eval_dataset = eval_dataset["validation"]

train_dataset = train_dataset.take(10000)
eval_dataset = eval_dataset.take(1000)

training_args = TrainingArguments(
        optim = "adamw_bnb_8bit",
        fp16=True,
        # tf32=True,
        gradient_checkpointing = True,

        num_train_epochs = 1,
        learning_rate = 2e-4,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 8,

        seed = 99,
        output_dir = "./checkpoints",

        save_strategy = "steps",
        eval_strategy = "steps",

        save_steps = 0.1,
        eval_steps = 0.1,
        logging_steps = 0.1,
        load_best_model_at_end = True,
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
    trainer_stats = trainer.train()
