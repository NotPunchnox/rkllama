# img_encoder.py
import numpy as np
from src.classes import *
from src.format_utils import prepare_image
from src.model_utils import read_data_from_file
import logging

logger = logging.getLogger("rkllama.rknn")

# Some default values
IMAGE_HEIGHT = 392
IMAGE_WIDTH = 392
IMAGE_TOKEN_NUM = 196
EMBED_SIZE = 1536

class RKNN:

    def __init__(self, model_path: str, core_max: int = RKNN_NPU_CORE_ALL, base_domain_id = 0):

        logger.debug(f"Initializing RKNN model from {model_path} with cores: {core_max}")

        # Custom properties saved for reference
        self.model_path = model_path
        self.core_mask = core_max
        
        # Prepare model buffer
        model_bytes = read_data_from_file(model_path)
        model_len = len(model_bytes)
        buf = ctypes.create_string_buffer(model_bytes, model_len)
        

        # Initialization of the RKNN model
        self.ctx = RknnAppContext(base_domain_id)
        self.rknn_init = rknn_lib.rknn_init
        self.rknn_init.argtypes = [ctypes.POINTER(RKNNContext), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        self.rknn_init.restype = ctypes.c_int
        ret = self.rknn_init(ctypes.byref(self.ctx.rknn_ctx), ctypes.cast(buf, ctypes.c_void_p), model_len, 0, None)
        if ret < 0:
            raise RuntimeError(f"Failed to initialize RKNN model: {ret}")

        # Setting the NPU cores to work
        #core_mask = RKNN_NPU_CORE_0_1 if self.core_num == 2 else (RKNN_NPU_CORE_0_1_2 if self.core_num == 3 else RKNN_NPU_CORE_AUTO)
        self.rknn_set_core_mask = rknn_lib.rknn_set_core_mask
        self.rknn_set_core_mask.argtypes = [RKNNContext, ctypes.c_uint32]
        self.rknn_set_core_mask.restype = ctypes.c_int
        ret = self.rknn_set_core_mask(self.ctx.rknn_ctx, self.core_mask)
        if ret < 0:
            raise RuntimeError(f"rknn_set_core_mask failed: {ret}")


        # Inut output numbers query
        self.io_num = RKNNInputOutputNum()
        self.rknn_query = rknn_lib.rknn_query
        self.rknn_query.argtypes = [RKNNContext, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        self.rknn_query.restype = ctypes.c_int
        ret = self.rknn_query(self.ctx.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(self.io_num), ctypes.sizeof(self.io_num))
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_query(in/out num) failed: {ret}")
        self.ctx.io_num = self.io_num


        # Inut attributes query
        self.input_attrs = []
        for i in range(self.io_num.n_input):
            a = RKNNTensorAttr()
            a.index = i
            ret = self.rknn_query(self.ctx.rknn_ctx, RKNN_QUERY_INPUT_ATTR, ctypes.byref(a), ctypes.sizeof(a))
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_query(input attr) failed: {ret}")
            self.input_attrs.append(a)
        self.ctx.input_attrs = self.input_attrs
        
        
        # Output attributes query
        self.output_attrs = []
        for i in range(self.io_num.n_output):
            a = RKNNTensorAttr()
            a.index = i
            ret = self.rknn_query(self.ctx.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(a), ctypes.sizeof(a))
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_query(output attr) failed: {ret}")
            self.output_attrs.append(a)
        self.ctx.output_attrs = self.output_attrs


        # interpret shape
        # header: RKNN_TENSOR_NCHW = 0, RKNN_TENSOR_NHWC = 1
        if self.input_attrs[0].fmt == 0:  # NCHW
            self.ctx.model_channel = self.input_attrs[0].dims[1]
            self.ctx.model_height = self.input_attrs[0].dims[2]
            self.ctx.model_width = self.input_attrs[0].dims[3]
        else:
            self.ctx.model_height = self.input_attrs[0].dims[1]
            self.ctx.model_width = self.input_attrs[0].dims[2]
            self.ctx.model_channel = self.input_attrs[0].dims[3]

        # Setting default values if not set        
        self.height = self.ctx.model_height or IMAGE_HEIGHT
        self.width  = self.ctx.model_width or IMAGE_WIDTH
        self.channel = self.ctx.model_channel or 3


        # Other Function prototypes
        self.rknn_destroy = rknn_lib.rknn_destroy
        self.rknn_destroy.argtypes = [RKNNContext]
        self.rknn_destroy.restype = ctypes.c_int

        self.rknn_inputs_set = rknn_lib.rknn_inputs_set
        self.rknn_inputs_set.argtypes = [RKNNContext, ctypes.c_uint32, ctypes.POINTER(RKNNInput)]
        self.rknn_inputs_set.restype = ctypes.c_int

        self.rknn_run = rknn_lib.rknn_run
        self.rknn_run.argtypes = [RKNNContext, ctypes.c_void_p]
        self.rknn_run.restype = ctypes.c_int

        self.rknn_outputs_get = rknn_lib.rknn_outputs_get
        self.rknn_outputs_get.argtypes = [RKNNContext, ctypes.c_uint32, ctypes.POINTER(RKNNOutput), ctypes.c_void_p]
        self.rknn_outputs_get.restype = ctypes.c_int

        self.rknn_outputs_release = rknn_lib.rknn_outputs_release
        self.rknn_outputs_release.argtypes = [RKNNContext, ctypes.c_uint32, ctypes.POINTER(RKNNOutput)]
        self.rknn_outputs_release.restype = ctypes.c_int

    def run(self, image_path):
        
        logger.info(f"Running RKNN model on image: {image_path}")
        
        # Run the model
        prepared_image = prepare_image(image_path, self.width, self.height)
        out = self.run_imgenc(prepared_image)
        expected = IMAGE_TOKEN_NUM * EMBED_SIZE

        if out.size < expected:
            pad = np.zeros((expected,), dtype=np.float32)
            pad[:min(out.size, expected)] = out[:min(out.size, expected)]
            return pad
        
        # Return the output
        return out[:expected].astype(np.float32)


    def stop(self):
        if self.ctx.rknn_ctx and self.ctx.rknn_ctx.value != 0:
            self.rknn_destroy(self.ctx.rknn_ctx)
        # Delete all attributes to free memory
        for attr_name in list(self.__dict__.keys()):
            delattr(self, attr_name)


    def run_imgenc(self, img_data: np.ndarray) -> np.ndarray:
        if img_data.dtype != np.uint8:
            raise ValueError("img_data must be uint8")

        input_tensor = RKNNInput()
        input_tensor.index = 0
        input_tensor.type = 3  # RKNN_TENSOR_UINT8
        input_tensor.fmt = 1   # RKNN_TENSOR_NHWC
        input_tensor.size = img_data.size
        input_tensor.buf = img_data.ctypes.data_as(ctypes.c_void_p)

        ret = self.rknn_inputs_set(self.ctx.rknn_ctx, 1, ctypes.byref(input_tensor))
        if ret < 0:
            raise RuntimeError(f"rknn_inputs_set failed: {ret}")

        ret = self.rknn_run(self.ctx.rknn_ctx, None)
        if ret < 0:
            raise RuntimeError(f"rknn_run failed: {ret}")

        out = RKNNOutput()
        out.want_float = 1
        ret = self.rknn_outputs_get(self.ctx.rknn_ctx, 1, ctypes.byref(out), None)
        if ret < 0:
            raise RuntimeError(f"rknn_outputs_get failed: {ret}")

        n = out.size // 4
        float_p = ctypes.cast(out.buf, ctypes.POINTER(ctypes.c_float))
        arr = np.ctypeslib.as_array(float_p, shape=(n,))
        result = np.copy(arr).astype(np.float32)

        self.rknn_outputs_release(self.ctx.rknn_ctx, 1, ctypes.byref(out))
        return result
        
