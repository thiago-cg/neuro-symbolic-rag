"""Entity normalization for triple extraction."""

import re

# Common academic abbreviation aliases
_ALIASES: dict[str, str] = {
    "bert": "bert",
    "gpt": "gpt",
    "gpt-2": "gpt_2",
    "gpt-3": "gpt_3",
    "gpt-4": "gpt_4",
    "llm": "llm",
    "large language model": "llm",
    "large language models": "llm",
    "nlp": "nlp",
    "natural language processing": "nlp",
    "ml": "machine_learning",
    "machine learning": "machine_learning",
    "dl": "deep_learning",
    "deep learning": "deep_learning",
    "nn": "neural_network",
    "neural network": "neural_network",
    "neural networks": "neural_network",
    "rnn": "rnn",
    "recurrent neural network": "rnn",
    "lstm": "lstm",
    "long short-term memory": "lstm",
    "cnn": "cnn",
    "convolutional neural network": "cnn",
    "transformer": "transformer",
    "attention": "attention_mechanism",
    "self-attention": "self_attention",
    "elmo": "elmo",
    "word2vec": "word2vec",
    "glove": "glove",
    "rl": "reinforcement_learning",
    "reinforcement learning": "reinforcement_learning",
    "cv": "computer_vision",
    "computer vision": "computer_vision",
}


def normalize_entity(text: str) -> str:
    """Normalize an entity for use in Datalog/ASP facts.

    Steps:
    1. Lowercase
    2. Check aliases
    3. Remove special characters (keep alphanumeric, underscore, hyphen)
    4. Replace whitespace with underscore
    5. Strip leading/trailing underscores

    Args:
        text: Raw entity string from LLM

    Returns:
        Normalized string safe for ASP atoms
    """
    if not text:
        return ""

    normalized = text.lower().strip()

    # Check aliases first (longest match)
    if normalized in _ALIASES:
        return _ALIASES[normalized]

    # Remove special characters except hyphen and space
    normalized = re.sub(r"[^\w\s-]", "", normalized)

    # Collapse multiple spaces/hyphens
    normalized = re.sub(r"[\s-]+", "_", normalized)

    # Strip leading/trailing underscores
    normalized = normalized.strip("_")

    return normalized or "unknown"
