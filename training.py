import os
import pickle
import pymupdf4llm
import torch
from docx import Document
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

# --- CONFIGURATION ---
CORPUS_DIR = "./calvino_pdfs"
BASE_MODEL = "unsloth/meta-llama-3.1-8b-bnb-4bit" # Or Llama 4 Scout
APPLE_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./calvino_lora_adapter"
MAX_TRAIN_SEQ_LENGTH = 1536
CHUNK_SIZE = 900
CACHE_DIR = "./.cache"
CORPUS_CACHE_PATH = os.path.join(CACHE_DIR, "corpus.pkl")
DATASET_CACHE_PATH = os.path.join(CACHE_DIR, "dataset.pkl")
CACHE_VERSION = 1


def _build_source_fingerprint(directory):
    files = []
    for file in sorted(os.listdir(directory)):
        if not (file.endswith(".pdf") or file.endswith(".docx")):
            continue
        path = os.path.join(directory, file)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        files.append((file, stat.st_size, stat.st_mtime_ns))
    return {
        "cache_version": CACHE_VERSION,
        "chunk_size": CHUNK_SIZE,
        "files": files,
    }


def _load_cache(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        print(f"Ignoring unreadable cache {path}: {exc}")
        return None


def _save_cache(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)

# 1. EXTRACTION: Convert PDFs and DOCX to a clean Markdown corpus
def load_corpus(directory):
    fingerprint = _build_source_fingerprint(directory)
    cached = _load_cache(CORPUS_CACHE_PATH)
    if cached and cached.get("fingerprint") == fingerprint:
        print(f"Loaded cached corpus from {CORPUS_CACHE_PATH}")
        return cached.get("corpus", []), fingerprint

    corpus = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        try:
            if file.endswith(".pdf"):
                # PyMuPDF4LLM preserves layout/italics which are key for Calvino
                corpus.append(pymupdf4llm.to_markdown(path))
            elif file.endswith(".docx"):
                doc = Document(path)
                corpus.append("\n".join([p.text for p in doc.paragraphs]))
        except Exception as exc:
            # Keep ingestion running when one source file is malformed.
            print(f"Skipping unreadable file {file}: {exc}")
    _save_cache(
        CORPUS_CACHE_PATH,
        {
            "fingerprint": fingerprint,
            "corpus": corpus,
        },
    )
    print(f"Saved corpus cache to {CORPUS_CACHE_PATH}")
    return corpus, fingerprint

# 2. DATASET PREP: Create synthetic "Before & After" pairs
def prepare_dataset(documents):
    # We only have Calvino-like source text, not true parallel rewrite pairs.
    # Build a synthetic "plain-ish" input from each chunk so the conditioning
    # text varies per sample instead of being a single constant prompt.
    def _to_plain_proxy(text):
        plain = " ".join(text.split())
        plain = (
            plain.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("—", "-")
            .replace("–", "-")
        )
        return plain

    data = []
    for text in documents:
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i + CHUNK_SIZE]
            chunk = chunk.strip()
            if len(chunk) < 120:
                continue
            data.append({
                "instruction": "Rewrite the following in the style of Italo Calvino:",
                "input": _to_plain_proxy(chunk),
                "output": chunk
            })
    return data


def load_or_prepare_dataset(documents, fingerprint):
    cached = _load_cache(DATASET_CACHE_PATH)
    if cached and cached.get("fingerprint") == fingerprint:
        print(f"Loaded cached dataset from {DATASET_CACHE_PATH}")
        return cached.get("dataset", [])

    ds = prepare_dataset(documents)
    _save_cache(
        DATASET_CACHE_PATH,
        {
            "fingerprint": fingerprint,
            "dataset": ds,
        },
    )
    print(f"Saved dataset cache to {DATASET_CACHE_PATH}")
    return ds


class StyleTransferDataset(TorchDataset):
    def __init__(self, records, tokenizer, max_length=MAX_TRAIN_SEQ_LENGTH):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        prompt = (
            "Below is a sentance. Rewrite it with the lightness, exactitude, "
            "and geometric precision of Italo Calvino.\n\n"
            f"### Paragraph:\n{record['input']}\n\n### Calvino Version:\n"
        )
        target = record["output"].strip()
        full_text = prompt + target

        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in full_tokens.items()}
        item["labels"] = item["input_ids"].clone()
        prompt_len = min(prompt_tokens["input_ids"].shape[-1], item["labels"].shape[-1])
        item["labels"][:prompt_len] = -100
        item["labels"][item["attention_mask"] == 0] = -100
        return item

# 3. TRAINING: Unsloth LoRA Fine-Tuning
def train_style_transfer(dataset):
    def _format_example(example):
        return (
            "Below is a paragraph. Rewrite it with the lightness, exactitude, "
            "and geometric precision of Italo Calvino.\n\n"
            f"### Paragraph:\n{example['input']}\n\n### Calvino Version:\n{example['output']}"
        )

    interrupted = False
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = BASE_MODEL,
            max_seq_length = 2048,
            load_in_4bit = True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = 32, # Higher rank for complex styles like Calvino
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha = 32,
            lora_dropout = 0,
        )

        # Standard SFT (Supervised Fine-Tuning) setup...
        # [Insert standard Unsloth SFTTrainer boilerplate here]
        # For brevity, use the Unsloth 'train' function.
    except (NotImplementedError, ImportError):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(APPLE_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            APPLE_BASE_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        tokenized_ds = StyleTransferDataset(dataset, tokenizer, max_length=MAX_TRAIN_SEQ_LENGTH)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"{OUTPUT_DIR}_checkpoints",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                logging_steps=5,
                save_strategy="steps",
                save_steps=25,
                report_to=[],
                remove_unused_columns=False,
                optim="adafactor",
                auto_find_batch_size=True,
            ),
            train_dataset=tokenized_ds,
            data_collator=default_data_collator,
        )
        try:
            checkpoint_dir = f"{OUTPUT_DIR}_checkpoints"
            last_checkpoint = get_last_checkpoint(checkpoint_dir) if os.path.isdir(checkpoint_dir) else None
            if last_checkpoint:
                print(f"Resuming from checkpoint: {last_checkpoint}")
                trainer.train(resume_from_checkpoint=last_checkpoint)
            else:
                print("No checkpoint found; starting training from step 0.")
                trainer.train()
        except KeyboardInterrupt:
            interrupted = True
            print("\nTraining interrupted by user; saving current adapter state...")

    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user; saving current adapter state...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    if interrupted:
        print(f"Saved interrupted training state to {OUTPUT_DIR}.")

def main():
    documents, fingerprint = load_corpus(CORPUS_DIR)
    if not documents:
        raise RuntimeError(f"No readable .pdf or .docx files found in {CORPUS_DIR}")
    ds = load_or_prepare_dataset(documents, fingerprint)
    if not ds:
        raise RuntimeError("Dataset preparation produced 0 samples.")
    train_style_transfer(ds)


if __name__ == "__main__":
    main()