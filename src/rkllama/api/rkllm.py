import ctypes
from .classes import *
import logging, multiprocessing
import numpy as np

logger = logging.getLogger("rkllama.rkllm")

# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    def __init__(self, callback, model_path, model_dir, options=None, lora_model_path = None, prompt_cache_path = None, base_domain_id = 1):
        
        logger.debug(f"Initializing RKLLM model from {model_path} with options: {options}")

        # Custom Attributes
        self.format_schema = None
        self.format_type = None
        self.format_options = {}
        self.model_dir = model_dir
        
        # Multi Load model attributes
        self.base_domain_id = base_domain_id
        
        # Configure RKLLM parameters
        self.rkllm_param = RKLLMParam()
        self.rkllm_param.model_path = bytes(model_path, 'utf-8')
        self.rkllm_param.max_context_len =  int(float((options.get("num_ctx", rkllama.config.get("model", "default_num_ctx")))))
        self.rkllm_param.max_new_tokens = int(options.get("max_new_tokens", rkllama.config.get("model", "default_max_new_tokens")))
        self.rkllm_param.top_k = float(options.get("top_k", rkllama.config.get("model", "default_top_k")))
        self.rkllm_param.top_p = float(options.get("top_p", rkllama.config.get("model", "default_top_p")))
        self.rkllm_param.temperature = float(options.get("temperature", rkllama.config.get("model", "default_temperature")))
        self.rkllm_param.repeat_penalty = float(options.get("repeat_penalty", rkllama.config.get("model", "default_repeat_penalty")))
        self.rkllm_param.frequency_penalty = float(options.get("frequency_penalty", rkllama.config.get("model", "default_frequency_penalty")))
        self.rkllm_param.presence_penalty = float(options.get("presence_penalty", rkllama.config.get("model", "default_presence_penalty")))
        self.rkllm_param.mirostat = int(options.get("mirostat", rkllama.config.get("model", "default_mirostat")))
        self.rkllm_param.mirostat_tau = float(options.get("mirostat_tau", rkllama.config.get("model", "default_mirostat_tau")))
        self.rkllm_param.mirostat_eta = float(options.get("mirostat_eta", rkllama.config.get("model", "default_mirostat_eta")))

        # Fixme: these parameters are not used in the current implementation, but they are set to default values
        self.rkllm_param.skip_special_token = True
        self.rkllm_param.n_keep = 0
        self.rkllm_param.is_async = False
        self.rkllm_param.use_gpu = True

        # For Image MultiModal Models
        self.rkllm_param.img_start = options.get("img_start", "").encode("utf-8");
        self.rkllm_param.img_end = options.get("img_end", "").encode("utf-8");
        self.rkllm_param.img_content = options.get("img_content", "").encode("utf-8");
        
        # Extend parameters for RKLLM
        self.rkllm_param.extend_param.base_domain_id = self.base_domain_id
        self.rkllm_param.extend_param.embed_flash = 1
        self.rkllm_param.extend_param.n_batch = 1
        self.rkllm_param.extend_param.use_cross_attn = 0
        #self.rkllm_param.extend_param.enabled_cpus_num = multiprocessing.cpu_count()
        self.rkllm_param.extend_param.enabled_cpus_num = 4  
        #self.rkllm_param.extend_param.enabled_cpus_mask = (1<<(self.rkllm_param.extend_param.enabled_cpus_num+1))-1
        processor = rkllama.config.get("platform", "processor", None)
        if processor.lower() in ["rk3576", "rk3588"]: # Recommended new way by Rockchip
            self.rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)
        else:
            self.rkllm_param.extend_param.enabled_cpus_mask = (1 << 0)|(1 << 1)|(1 << 2)|(1 << 3)
        
        
        # Initialization of the RKLLM model
        self.handle = RKLLM_Handle_t()
        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(self.rkllm_param), callback)
        if(ret != 0):
            raise RuntimeError(f"Failed to initialize RKLLM model: {ret}")

        
        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_chat_template.restype = ctypes.c_int
        
        self.set_function_tools_ = rkllm_lib.rkllm_set_function_tools
        self.set_function_tools_.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_function_tools_.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.rkllm_clear_kv_cache = rkllm_lib.rkllm_clear_kv_cache
        self.rkllm_clear_kv_cache.argtypes = [RKLLM_Handle_t, ctypes.c_int , ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int) ]
        self.rkllm_clear_kv_cache.restype = ctypes.c_int

        self.rkllm_abort = rkllm_lib.rkllm_abort

        # CHeck if chat template options are provided
        if all(item in options for item in ["system_prompt","prompt_prefix","prompt_postfix"]): 
            system_prompt = options.get("system_prompt")
            prompt_prefix = options.get("prompt_prefix")
            prompt_postfix = options.get("prompt_postfix")
            self.set_chat_template(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(prompt_prefix.encode('utf-8')), ctypes.c_char_p(prompt_postfix.encode('utf-8')))

        self.lora_adapter_path = None
        self.lora_model_name = None
        self.rkllm_lora_params = None
        if lora_model_path:
            self.lora_adapter_path = lora_model_path
            self.lora_adapter_name = "test"

            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((self.lora_adapter_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((self.lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((self.lora_adapter_name).encode('utf-8'))
        

        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.lora_params = ctypes.pointer(self.rkllm_lora_params) if self.rkllm_lora_params else None
        self.rkllm_infer_params.keep_history = 0


        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))
        
        self.tools = None

    def tokens_to_ctypes_array(self, tokens, ctype):
        return (ctype * len(tokens))(*tokens)

    def set_function_tools(self, system_prompt, tools, tool_response_str):
        if self.tools is None or not self.tools == tools:
            self.tools = tools
            self.set_function_tools_(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(tools.encode('utf-8')),  ctypes.c_char_p(tool_response_str.encode('utf-8')))

    def run(self, *param):
        
        # Get the arguments
        inference_mode, model_input_type, input = param

        # Define the input object
        rkllm_input = RKLLMInput()
        rkllm_input.input_type = model_input_type

        # Set the inference mode
        self.rkllm_infer_params.mode = inference_mode
        
        # CHeck the model type to construct parameters
        if model_input_type == RKLLMInputType.RKLLM_INPUT_TOKEN:
            token_input = input
            if token_input[-1] != 2:  
                token_input.append(2)
            token_array = (ctypes.c_int * len(token_input))(*token_input)
            
            rkllm_input.input_data.token_input.input_ids = ctypes.cast(token_array, ctypes.POINTER(ctypes.c_int32))
            rkllm_input.input_data.token_input.n_tokens = ctypes.c_size_t(len(token_input))
            
        elif model_input_type == RKLLMInputType.RKLLM_INPUT_EMBED:
            embed_input = input
            num_tokens, embd_size = embed_input.shape

            flat = embed_input.ravel().astype(np.float32)
            embed_array = (ctypes.c_float * flat.size)(*flat)
 
            rkllm_input.input_data.embed_input.embed = ctypes.cast(embed_array, ctypes.POINTER(ctypes.c_float))
            rkllm_input.input_data.embed_input.n_tokens = ctypes.c_size_t(num_tokens)
        
        elif model_input_type == RKLLMInputType.RKLLM_INPUT_MULTIMODAL:
            prompt_input, image_embed, n_image_tokens, image_width, image_height, num_images = input
            logger.debug(f"Running multimodal inference with {num_images} images, each of size {image_width}x{image_height}, and {n_image_tokens} image tokens.")
            
            # Prompt
            rkllm_input.input_data.multimodal_input.prompt = prompt_input.encode("utf-8")
            
            # Image Embedding
            arr = image_embed.flatten().astype(np.float32)
            rkllm_input.input_data.multimodal_input.image_embed = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            rkllm_input.input_data.multimodal_input.n_image_tokens = ctypes.c_size_t(n_image_tokens)
            rkllm_input.input_data.multimodal_input.n_image = ctypes.c_size_t(num_images)
            rkllm_input.input_data.multimodal_input.image_width = ctypes.c_size_t(image_width)
            rkllm_input.input_data.multimodal_input.image_height = ctypes.c_size_t(image_height)
        

        # Run the RKLLM model with the input
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
        return

    def abort(self):
        return self.rkllm_abort(self.handle)
    
    def clear_cache(self):
        return self.rkllm_clear_kv_cache(self.handle, 1, None, None)

    def release(self):
        self.rkllm_destroy(self.handle)
