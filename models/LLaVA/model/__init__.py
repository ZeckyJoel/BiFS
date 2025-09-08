import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_gemma": "LlavaGemmaForCausalLM, LlavaGemmaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # Add other models as needed
    "llava_llama_more_picture": "LlavaLlamaForCausalLM, LlavaConfig",
    #"llava_llama_withblip": "LlavaLlamaForCausalLM, LlavaConfig",
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except ImportError:
        # import traceback
        # traceback.print_exc()
        print(f"Failed to import {model_name} from llava.language_model.{model_name}")
        pass
