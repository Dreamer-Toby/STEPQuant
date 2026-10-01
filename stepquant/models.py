"""Checkpoint loading stays optional; the mathematical core only needs PyTorch."""
import json
from pathlib import Path
import torch
from unittest.mock import patch


def load_model(path, max_memory_gib=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = json.loads((Path(path) / 'config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    kwargs = dict(local_files_only=True, torch_dtype=torch.bfloat16,
                  attn_implementation='eager')
    devices = torch.cuda.device_count()
    if devices:
        kwargs['device_map'] = 'auto'
        if max_memory_gib is not None:
            kwargs['max_memory'] = {i: f'{max_memory_gib}GiB' for i in range(devices)}
    if config['model_type'] == 'qwen3_5':
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        text_config = Qwen3_5TextConfig(**config['text_config'])
        model = Qwen3_5ForCausalLM.from_pretrained(path, config=text_config, **kwargs)
    elif config['model_type'] == 'kimi_linear':
        # Transformers 4.57.1's documentation decorator mishandles PEP-604 unions
        # in this checkpoint. Skip doc generation only during import, then restore it.
        with patch('transformers.utils.auto_docstring', lambda fn: fn):
            model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, **kwargs)
        # The checkpoint constructor forces FlashAttention even when eager was
        # requested. Its own eager attention is sufficient for this reference path.
        model.config._attn_implementation = 'eager'
        model.model._use_flash_attention_2 = False
    else:
        raise ValueError(f"unsupported model_type: {config['model_type']}")
    return model.eval(), tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=32):
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    device = model.get_input_embeddings().weight.device
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    cache, generated = None, []
    stop = tokenizer.eos_token_id
    stops = set(stop if isinstance(stop, list) else [stop])
    for _ in range(max_new_tokens):
        output = model(input_ids=ids, past_key_values=cache, use_cache=True, **last_logits_kwargs(model))
        cache = output.past_key_values
        ids = output.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(int(ids.item()))
        if generated[-1] in stops:
            break
    return tokenizer.decode(generated, skip_special_tokens=True), generated


def last_logits_kwargs(model):
    return {'generation_mode': True} if model.config.model_type == 'kimi_linear' else {'logits_to_keep': 1}
