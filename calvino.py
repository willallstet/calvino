import re
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = "./calvino_lora_adapter"
HARDCODED_INPUT = ("Consistency indicates a certain well-formed completeness: that the systems within the text borrow from the rules in such a way that their operation is consistent. A consistent piece of writing does not cheat out or double back in making its point. Well-formed, complete and cohesive as the defining metrics of consistency stand in opposition to a more conventional view of the word. It is not unchanging, but rather it is stable in a different sense. It is an arena without disjoint sections in which different, diverse, and multiple operations are able to play out. Think of the consistency of mathematical axioms that provide the stage for an entire discursive field rather than the predictable storylines of a daytime soap opera. In Melville’s Bartleby, The Scrivener, Bartleby utilizes the vastness of language’s completeness of language to draw out the compelling tensions that arise when consistency is respected. With “I would prefer not to” he is able to nullify the narrator’s while remaining completely, impressively deftly, consistent. There is no elaborate, rule-breaking conceit to create narrative conflict. Instead, it is born from within the structures of language. Here, like in a Brontë novel, the two conditions are in delicate tension with each other, yet consistent. It is this sensitive attenuation to rules that creates a strong narrative and syntactic interest. I think of writers like Pynchon who draw it to extremes as other examples of this consistent, rule-based textual and narrative tension. A good writer is able to maintain their consistency while simultaneously drawing out tension from within it. No cheating out narrative or linguistic structures to create a deus (or diabolus) ex machina."
)


def _best_dtype(device: str) -> torch.dtype:
    # fp16 is typically the best memory/perf trade-off for local inference.
    if device in {"mps", "cuda"}:
        return torch.float16
    return torch.float32


def _split_into_sentences(text: str):
    # Keep punctuation with each sentence and avoid empty chunks.
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in pieces if part.strip()]


def _token_set(text: str):
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def _lexical_overlap(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens.intersection(b_tokens)) / max(len(a_tokens), 1)


def _rewrite_is_usable(original: str, rewritten: str) -> bool:
    if not rewritten:
        return False
    ratio = len(rewritten) / max(len(original), 1)
    if ratio < 0.55 or ratio > 1.9:
        return False
    if _lexical_overlap(original, rewritten) < 0.2:
        return False
    if "|" in rewritten or "http://" in rewritten or "https://" in rewritten:
        return False
    return True


def rewrite_in_calvino_style(user_paragraph):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = _best_dtype(device)

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=OUTPUT_DIR,  # Load your trained adapter
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except (NotImplementedError, ImportError):
        peft_config = PeftConfig.from_pretrained(OUTPUT_DIR)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    rewritten_sentences = []
    sentences = _split_into_sentences(user_paragraph)
    if not sentences:
        sentences = [user_paragraph.strip()]

    for sentence in sentences:
        prompt = f"""Rewrite this sentence in the style of Italo Calvino.
        Preserve the concrete meaning and key details exactly.
        Return exactly one sentence.
        
        Sentence:
        {sentence}
        
        Rewritten sentence:"""

        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, input_len:]
        rewritten = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated_sentences = _split_into_sentences(rewritten)
        if generated_sentences:
            candidate = generated_sentences[0]
            if _rewrite_is_usable(sentence, candidate):
                rewritten_sentences.append(candidate)
                continue
        rewritten_sentences.append(sentence)

    return " ".join(rewritten_sentences).strip()


if __name__ == "__main__":
    result = rewrite_in_calvino_style(HARDCODED_INPUT)
    print(result)