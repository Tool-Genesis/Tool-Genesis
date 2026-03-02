def get_tokenizer(model):
    model_mapping = {
        "openai/gpt-4o": "gpt-4o",
        "openai/gpt-4": "gpt-4",
        "openai/gpt-4-turbo": "gpt-4-turbo",
        "openai/gpt-3.5-turbo": "gpt-3.5-turbo",
        "anthropic/claude-3-sonnet": "gpt-4",
        "anthropic/claude-3-haiku": "gpt-4",
        "anthropic/claude-3-opus": "gpt-4",
    }
    try:
        m = (model or "").lower() if isinstance(model, str) else ""
        if ("llama" in m) or ("meta-llama" in m):
            return tiktoken.get_encoding("cl100k_base")
        tiktoken_model = model_mapping.get(model, model)
        return tiktoken.encoding_for_model(tiktoken_model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text, model=None):
    """ Get the number of tokens for a string, measured using tiktoken. """

    try:
        model = model or "gpt-4o"
        if isinstance(model, str):
            tokenizer = get_tokenizer(model)
        else:
            # Assume model is a namedtuple (model_name, tokenizer)
            tokenizer = model.tokenizer
    except:
        model = "gpt-4o"
        if isinstance(model, str):
            tokenizer = get_tokenizer(model)
        else:
            # Assume model is a namedtuple (model_name, tokenizer)
            tokenizer = model.tokenizer
        

    return len(tokenizer.encode(text))
