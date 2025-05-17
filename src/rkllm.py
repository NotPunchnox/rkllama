import ctypes
from .classes import *
from .callback import *

# Connect the callback function between Python and C++
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    def __init__(self, model_path, model_dir, temperature=0.8, context_length=2048, lora_model_path = None, prompt_cache_path = None):
        
        self.format_schema = None
        self.format_type = None
        self.format_options = {}
        self.model_dir = model_dir
        
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = context_length
        rkllm_param.max_new_tokens = -1
        rkllm_param.skip_special_token = True

        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = temperature
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        rkllm_param.is_async = False

        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        
        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)
        if(ret != 0):
            raise RuntimeError(f"Failed to initialize RKLLM model: {ret}")

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.lora_adapter_path = None
        self.lora_model_name = None
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
        
        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def tokens_to_ctypes_array(self, tokens, ctype):
        return (ctype * len(tokens))(*tokens)

    def run(self, prompt_tokens):
        rkllm_lora_params = None
        if self.lora_model_name:
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((self.lora_model_name).encode('utf-8'))
        
        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        rkllm_infer_params.lora_params = ctypes.byref(rkllm_lora_params) if rkllm_lora_params else None

        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_TOKEN

        if prompt_tokens[-1] != 2:  
            prompt_tokens.append(2)

        token_array = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)

        rkllm_input.input_data.token_input.input_ids = token_array
        rkllm_input.input_data.token_input.n_tokens = ctypes.c_ulong(len(prompt_tokens))


        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)

        return

    def release(self):
        self.rkllm_destroy(self.handle)
