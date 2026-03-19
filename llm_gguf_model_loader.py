import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import logging
from .utils import get_llm_ggufs, get_llm_gguf_path

logger = logging.getLogger("LLM-SDXL-Adapter")


class LLMGGUFModelLoader:
    """
    ComfyUI node that loads Language Model and tokenizer
    Supports various LLM architectures (Gemma, Llama, Mistral, etc.)
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_llm_ggufs(), {
                    "default": get_llm_ggufs()[0] if get_llm_ggufs() else None
                }),
            },
            "optional": {
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {
                    "default": "auto"
                }),
                "force_reload": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_MODEL", "LLM_TOKENIZER", "STRING")
    RETURN_NAMES = ("model", "tokenizer", "info")
    FUNCTION = "load_model"
    CATEGORY = "llm_sdxl"
    
    def load_model(self, model_name, device="auto", force_reload=False):
        """Load Language Model and tokenizer"""
        if device == "auto":
            device = self.device
                
        try:
            model_path = get_llm_gguf_path(model_name)

            # Check if we need to reload
            if force_reload or self.model is None or self.current_model_path != model_path:
                # Clear previous model
                if self.model is not None:
                    del self.model
                    del self.tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                
                logger.info(f"Loading standalone GGUF {model_name} from {model_path}")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    gguf_file = model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    output_hidden_states=True,
                    trust_remote_code=True
                )
                
                # Try loading directly from the binary metadata first
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        gguf_file=model_name,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except Exception:
                    logger.warning("Internal GGUF tokenizer not found. Checking local folder...")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    except Exception as tok_err:
                        error_msg = (
                            f"FATAL: Could not find a tokenizer for {model_name}.\n"
                            f"GGUF internal check failed, and no tokenizer files found in {model_path}.\n"
                            "Please ensure tokenizer_config.json is in the model folder."
                        )
                        logger.error(error_msg)
                        raise FileNotFoundError(error_msg)
                
                self.current_model_path = model_path
                logger.info("GGUF and Tokenizer loaded successfully from local source.")
            
            info = f"Model: {model_path}\nDevice: {device}\nLoaded: {self.model is not None}"
            
            return (self.model, self.tokenizer, info)
            
        except Exception as e:
            logger.error(f"GGUF Load Failed: {str(e)}")
            raise e



# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMGGUFModelLoader": LLMGGUFModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMGGUFModelLoader": "LLM GGUF Model Loader",
} 