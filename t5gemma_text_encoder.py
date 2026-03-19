import torch
import gc
import logging

logger = logging.getLogger("LLM-SDXL-Adapter")


class T5GEMMATextEncoder:
    """
    ComfyUI node that encodes text using a loaded Language Model
    Supports various LLM architectures with chat templates
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_tokenizer": ("LLM_TOKENIZER",),
                "text": ("STRING", {"multiline": True, "default": "masterpiece, best quality, 1girl, anime style"}),
                "max_length": ("INT", {"default": 512, "min": 8, "max": 4096}),
                "device": (["cpu", "mps", "cuda"], {"default": "cuda"}),
                "dtype": (["float32", "bfloat16"], {"default": "bfloat16"}),
            }
        }
    
    RETURN_TYPES = ("LLM_HIDDEN_STATES", "LLM_ATTENTION_MASK", "STRING")
    RETURN_NAMES = ("hidden_states", "attention_mask", "info")
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl"
    
    def encode_text(self, llm_model, llm_tokenizer, text, max_length, device, dtype):
        """
        Encode text using Language Model and return hidden states
        """
        try:
            target_device = torch.device(device)
            target_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

            llm_model.to(target_device).to(target_dtype)

            # Tokenize
            inputs = llm_tokenizer(
                text + "<eos>",
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )

            input_ids = inputs.input_ids.to(target_device).contiguous()
            attention_mask = inputs.attention_mask.to(target_device).contiguous()

            # Generate hidden states
            with torch.no_grad():
                if device == "mps":
                    torch.mps.synchronize()

                outputs = llm_model(input_ids=input_ids, attention_mask=attention_mask)
                # Extract hidden states
                hidden_states = outputs.last_hidden_state.detach().contiguous()

            del input_ids, outputs

            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            
            # Prepare info
            info = f"Text: {text[:50]}...\nEncoded: {hidden_states.shape[1]}\nShape: {hidden_states.shape}"
            
            logger.info(f"Encoded text with shape: {hidden_states.shape}")
            
            return (hidden_states, attention_mask.to(torch.int), info)
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise Exception(f"Text encoding failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "T5GEMMATextEncoder": T5GEMMATextEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMATextEncoder": "T5Gemma Text Encoder"
} 