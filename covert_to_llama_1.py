import math
import os
import shutil
import sys
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM)



def rebalance_weights2(model_path, method):
    target_model_path = model_path + "-" + method
    shutil.copytree(model_path, target_model_path, dirs_exist_ok=True)  # copy includes optimizer

    source_model = AutoModelForCausalLM.from_pretrained(target_model_path, trust_remote_code=True)

    if os.path.exists(target_model_path + "/model.safetensors"):
        os.remove(target_model_path + "/model.safetensors")  # prepare for save_pretrained

    if os.path.exists(target_model_path + "/model.safetensors.index.json"):
        os.remove(target_model_path + "/model.safetensors.index.json")
        os.remove(target_model_path + "/model-00001-of-00002.safetensors")
        os.remove(target_model_path + "/model-00002-of-00002.safetensors")

    target_config = LlamaConfig(
        attention_bias=True,
        attention_dropout=source_model.config.attention_dropout,
        bos_token_id=source_model.config.bos_token_id,
        eos_token_id=source_model.config.eos_token_id,
        head_dim=source_model.config.hidden_size // source_model.config.num_attention_heads,
        hidden_act=source_model.config.hidden_act,
        hidden_size=source_model.config.hidden_size,
        initializer_range=source_model.config.initializer_range,
        intermediate_size=source_model.config.intermediate_size,
        max_position_embeddings=source_model.config.max_position_embeddings,
        mlp_bias=False,
        num_attention_heads=source_model.config.num_attention_heads,
        num_hidden_layers=source_model.config.num_hidden_layers,
        num_key_value_heads=source_model.config.num_key_value_heads,
        pretraining_tp=1,
        rms_norm_eps=source_model.config.rms_norm_eps,
        rope_scaling=None,
        rope_theta=source_model.config.rope_theta,
        tie_word_embeddings=False,
        torch_dtype=torch.float32,
        use_cache=True,
        vocab_size=source_model.config.vocab_size,
    )

    state_dict = source_model.state_dict()
    state_dict["model.embed_tokens.weight"] = state_dict["model.embed_tokens.weight"] * source_model.config.scale_emb
    for i in range(source_model.config.num_hidden_layers):
        state_dict[f"model.layers.{i}.self_attn.o_proj.bias"] = torch.zeros((source_model.config.hidden_size,), dtype=state_dict[f"model.layers.{i}.mlp.down_proj.weight"].dtype)
        state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] * source_model.config.scale_depth / math.sqrt(source_model.config.num_hidden_layers)
        state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = state_dict[f"model.layers.{i}.mlp.down_proj.weight"] * source_model.config.scale_depth / math.sqrt(source_model.config.num_hidden_layers)

    target_model = LlamaForCausalLM(target_config)
    # target_model = source_model
    target_model.load_state_dict(state_dict)

    target_model = target_model.to(torch.bfloat16)
    target_model.save_pretrained(target_model_path)
    print(target_model_path)

if __name__ == "__main__":
    rebalance_weights2(sys.argv[1], method="llama")