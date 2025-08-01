import ctypes
from .classes import *
from .callback import *
import logging, multiprocessing

logger = logging.getLogger("rkllama.rkllm")

# Connect the callback function between Python and C++
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    def __init__(self, model_path, model_dir, options=None, lora_model_path = None, prompt_cache_path = None):
        
        logger.debug(f"Initializing RKLLM model from {model_path} with options: {options}")

        # Custom Attributes
        self.format_schema = None
        self.format_type = None
        self.format_options = {}
        self.model_dir = model_dir
        
        # Configure RKLLM parameters
        self.rkllm_param = RKLLMParam()
        self.rkllm_param.model_path = bytes(model_path, 'utf-8')
        self.rkllm_param.max_context_len = int(options.get("num_ctx", config.get("model", "default_num_ctx")))
        self.rkllm_param.max_new_tokens = int(options.get("max_new_tokens", config.get("model", "default_max_new_tokens")))
        self.rkllm_param.top_k = int(options.get("top_k", config.get("model", "default_top_k")))
        self.rkllm_param.top_p = float(options.get("top_p", config.get("model", "default_top_p")))
        self.rkllm_param.temperature = float(options.get("temperature", config.get("model", "default_temperature")))
        self.rkllm_param.repeat_penalty = float(options.get("repeat_penalty", config.get("model", "default_repeat_penalty")))
        self.rkllm_param.frequency_penalty = float(options.get("frequency_penalty", config.get("model", "default_frequency_penalty")))
        self.rkllm_param.presence_penalty = float(options.get("presence_penalty", config.get("model", "default_presence_penalty")))
        self.rkllm_param.mirostat = int(options.get("mirostat", config.get("model", "default_mirostat")))
        self.rkllm_param.mirostat_tau = float(options.get("mirostat_tau", config.get("model", "default_mirostat_tau")))
        self.rkllm_param.mirostat_eta = float(options.get("mirostat_eta", config.get("model", "default_mirostat_eta")))

        # Fixme: these parameters are not used in the current implementation, but they are set to default values
        self.rkllm_param.skip_special_token = True
        self.rkllm_param.n_keep = 0
        self.rkllm_param.is_async = False

        # For Image MultiModal Models
        self.rkllm_param.img_start = "".encode('utf-8')
        self.rkllm_param.img_end = "".encode('utf-8')
        self.rkllm_param.img_content = "".encode('utf-8')

        # Extend parameters for RKLLM
        self.rkllm_param.extend_param.base_domain_id = 0
        self.rkllm_param.extend_param.embed_flash = 1
        self.rkllm_param.extend_param.n_batch = 1
        self.rkllm_param.extend_param.use_cross_attn = 0
        self.rkllm_param.extend_param.enabled_cpus_num = multiprocessing.cpu_count()
        self.rkllm_param.extend_param.enabled_cpus_mask = (1<<(self.rkllm_param.extend_param.enabled_cpus_num+1))-1
        #self.rkllm_param.extend_param.enabled_cpus_num = 4                                     # Better for RK3588
        #self.rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)  # Better for RK3588
        
        
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

        self.rkllm_abort = rkllm_lib.rkllm_abort

        # system_prompt = "<|im_start|>system You are a helpful assistant. <|im_end|>"
        # prompt_prefix = "<|im_start|>user"
        # prompt_postfix = "<|im_end|><|im_start|>assistant"
        # self.set_chat_template(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(prompt_prefix.encode('utf-8')), ctypes.c_char_p(prompt_postfix.encode('utf-8')))

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
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
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

    def run(self, prompt_tokens):
        if prompt_tokens[-1] != 2:  
            prompt_tokens.append(2)
        token_array = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)
        
        rkllm_input = RKLLMInput()
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_TOKEN
        rkllm_input.input_data.token_input.input_ids = ctypes.cast(token_array, ctypes.POINTER(ctypes.c_int32))
        rkllm_input.input_data.token_input.n_tokens = ctypes.c_size_t(len(prompt_tokens))
        
        # Run the RKLLM model with the input
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
        return

    def abort(self):
        return self.rkllm_abort(self.handle)
    
    def release(self):
        self.rkllm_destroy(self.handle)

        
