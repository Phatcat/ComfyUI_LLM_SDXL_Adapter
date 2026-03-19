import torch
import logging

logger = logging.getLogger("LLM-SDXL-Adapter")

class ApplyLLMToSDXLAdapter:
    """
    ComfyUI node that applies loaded LLM to SDXL adapter transformation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_hidden_states": ("LLM_HIDDEN_STATES",),
                "llm_adapter": ("LLM_ADAPTER",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "apply_adapter"
    CATEGORY = "llm_sdxl"
    
    def apply_adapter(self, llm_hidden_states, llm_adapter):
        """Apply the LLM to SDXL adapter transformation"""
        try:
            device = next(llm_adapter.parameters()).device
            dtype = next(llm_adapter.parameters()).dtype
            
            if llm_hidden_states.device != device or llm_hidden_states.dtype != dtype:
                llm_hidden_states = llm_hidden_states.to(device).to(dtype).contiguous()
            
            # Apply adapter
            with torch.no_grad():
                conditioning, pooled_output = llm_adapter(llm_hidden_states)
                del llm_hidden_states
            
            # Move tensors to CPU for ComfyUI conditioning system
            conditioning = conditioning.detach().cpu().contiguous()
            pooled_output = pooled_output.detach().cpu().contiguous()
            
            # Garbage collection
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            
            # Format conditioning for ComfyUI
            # ComfyUI expects conditioning as a list of [cond_tensor, metadata_dict] tuples
            comfy_conditioning = [[conditioning, {"pooled_output": pooled_output}]]
            
            # Prepare info
            info = f"Conditioning shape: {conditioning.shape}"
            
            logger.info(f"Applied LLM to SDXL adapter: {info}")
            
            return (comfy_conditioning, info)
            
        except Exception as e:
            logger.error(f"Failed to apply adapter: {str(e)}")
            raise Exception(f"Adapter application failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "ApplyLLMToSDXLAdapter": ApplyLLMToSDXLAdapter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyLLMToSDXLAdapter": "Apply LLM To SDXL Adapter"
} 
