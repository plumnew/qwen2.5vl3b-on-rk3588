from typing import Optional, Union, List, Callable, Dict, Any
import ctypes
from ctypes import Structure, Union, POINTER, CFUNCTYPE, c_void_p, c_char_p, c_int8, c_int32, c_uint32, c_uint8, c_float, c_size_t, c_bool
import numpy as np

# CPU constants
CPU0 = 1 << 0
CPU1 = 1 << 1
CPU2 = 1 << 2
CPU3 = 1 << 3
CPU4 = 1 << 4
CPU5 = 1 << 5
CPU6 = 1 << 6
CPU7 = 1 << 7
# State enums
class State:
    NORMAL = 0
    WAITING = 1
    FINISH = 2
    ERROR = 3
# Input types
class InputType:
    PROMPT = 0
    TOKEN = 1
    EMBED = 2
    MULTIMODAL = 3
# Inference modes
class InferMode:
    GENERATE = 0
    LAST_HIDDEN_LAYER = 1
    LOGITS = 2
# Internal ctypes structures (hidden from users)
class _RKLLMExtendParam(Structure):
    _fields_ = [
        ("base_domain_id", c_int32),
        ("embed_flash", c_int8),
        ("enabled_cpus_num", c_int8),
        ("enabled_cpus_mask", c_uint32),
        ("reserved", c_uint8 * 106)
    ]
class _RKLLMLoraAdapter(Structure):
    _fields_ = [
        ("lora_adapter_path", c_char_p),
        ("lora_adapter_name", c_char_p),
        ("scale", c_float)
    ]

class RKLLMMultiModelInput:
    def __init__(self):
        self.prompt: str = ""
        self.image_embed: np.ndarray  = None
        self.n_image_tokens: int = 196
        self.n_image: int = 1
        self.image_width: int = 392
        self.image_height: int = 392
    
class _RKLLMLoraParam(Structure):
    _fields_ = [
        ("lora_adapter_name", c_char_p)
    ]
class _RKLLMPromptCacheParam(Structure):
    _fields_ = [
        ("save_prompt_cache", c_int32),
        ("prompt_cache_path", c_char_p)
    ]
class _RKLLMInferParam(Structure):
    _fields_ = [
        ("mode", c_int32),
        ("lora_params", POINTER('_RKLLMLoraParam')),
        ("prompt_cache_params", POINTER('_RKLLMPromptCacheParam')),
        ("keep_history", c_int32)
    ]

class RKLLMLoraParam:
    def __init__(self):
        self.lora_adapter_name: str = ""

class RKLLMPromptCacheParam:
    def __init__(self):
        self.save_prompt_cache: int = 0
        self.prompt_cache_path: str = ""

class RKLLMInferParam:
    def __init__(self):
        self.mode: int = InferMode.GENERATE  # Default to GENERATE mode
        self.lora_params: RKLLMLoraParam = RKLLMLoraParam()
        self.prompt_cache_params: RKLLMPromptCacheParam = RKLLMPromptCacheParam()
        self.keep_history: int = 1
        
class _RKLLMResultLastHiddenLayer(Structure):
    _fields_ = [
        ("hidden_states", POINTER(c_float)),
        ("embd_size", c_int32),
        ("num_tokens", c_int32)
    ]

class _RKLLMResultLogits(Structure):
    _fields_ = [
        ("logits", POINTER(c_float)),
        ("vocab_size", c_int32),
        ("num_tokens", c_int32)
    ]

class _RKLLMResult(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("token_id", c_int32),
        ("last_hidden_layer", _RKLLMResultLastHiddenLayer),
        ("logits", _RKLLMResultLogits)
    ]

class _RKLLMEmbedInput(Structure):
    _fields_ = [
        ("embed", POINTER(c_float)),
        ("n_tokens", c_size_t)
    ]
class _RKLLMTokenInput(Structure):
    _fields_ = [
        ("input_ids", POINTER(c_int32)),
        ("n_tokens", c_size_t)
    ]
class _RKLLMMultiModelInput(Structure):
    _fields_ = [
        ("prompt", c_char_p),
        ("image_embed", POINTER(c_float)),
        ("n_image_tokens", c_size_t),
        ("n_image", c_size_t),
        ("image_width", c_size_t),
        ("image_height", c_size_t)
    ]
class _InputUnion(Union):
    _fields_ = [
        ("prompt_input", c_char_p),
        ("embed_input", _RKLLMEmbedInput),
        ("token_input", _RKLLMTokenInput),
        ("multimodal_input", _RKLLMMultiModelInput)
    ]
class _RKLLMInput(Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [
        ("input_mode", c_int32),
        ("input_data", _InputUnion)
    ]

class _RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", _RKLLMExtendParam),
    ]
_LLMResultCallback = CFUNCTYPE(None, POINTER(_RKLLMResult), c_void_p, c_int32)

class RKLLM:
    def __init__(self):
        self._handle = c_void_p(None)
        self._load_library()
        self._setup_functions()
        self._current_callback = None
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("RKLLM exit")
        self.destroy()

    def _load_library(self):
        self._lib = ctypes.CDLL("librkllmrt.so")
        if not self._lib:
            raise RuntimeError("Failed to load librkllmrt.so")

    def _setup_functions(self):
        self._lib.rkllm_createDefaultParam.restype = _RKLLMParam
        self._lib.rkllm_createDefaultParam.argtypes = []

        self._lib.rkllm_init.restype = c_int32
        self._lib.rkllm_init.argtypes = [POINTER(c_void_p), POINTER(_RKLLMParam), _LLMResultCallback]

        self._lib.rkllm_load_lora.restype = c_int32
        self._lib.rkllm_load_lora.argtypes = [c_void_p, POINTER(_RKLLMLoraAdapter)]

        self._lib.rkllm_load_prompt_cache.restype = c_int32
        self._lib.rkllm_load_prompt_cache.argtypes = [c_void_p, c_char_p]

        self._lib.rkllm_release_prompt_cache.restype = c_int32
        self._lib.rkllm_release_prompt_cache.argtypes = [c_void_p]

        self._lib.rkllm_destroy.restype = c_int32
        self._lib.rkllm_destroy.argtypes = [c_void_p]

        self._lib.rkllm_run.restype = c_int32
        self._lib.rkllm_run.argtypes = [c_void_p, POINTER(_RKLLMInput), POINTER(_RKLLMInferParam), c_void_p]

        self._lib.rkllm_run_async.restype = c_int32
        self._lib.rkllm_run_async.argtypes = [c_void_p, POINTER(_RKLLMInput), POINTER(_RKLLMInferParam), c_void_p]

        self._lib.rkllm_abort.restype = c_int32
        self._lib.rkllm_abort.argtypes = [c_void_p]

        self._lib.rkllm_is_running.restype = c_int32
        self._lib.rkllm_is_running.argtypes = [c_void_p]

        self._lib.rkllm_clear_kv_cache.restype = c_int32
        self._lib.rkllm_clear_kv_cache.argtypes = [c_void_p, c_int32]

        self._lib.rkllm_set_chat_template.restype = c_int32
        self._lib.rkllm_set_chat_template.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]

    def create_default_params(self) -> Dict[str, Any]:
        c_params = self._lib.rkllm_createDefaultParam()
        return {
            'model_path': None,
            'max_context_len': c_params.max_context_len,
            'max_new_tokens': c_params.max_new_tokens,
            'top_k': c_params.top_k,
            'n_keep': c_params.n_keep,
            'top_p': c_params.top_p,
            'temperature': c_params.temperature,
            'repeat_penalty': c_params.repeat_penalty,
            'frequency_penalty': c_params.frequency_penalty,
            'presence_penalty': c_params.presence_penalty,
            'mirostat': c_params.mirostat,
            'mirostat_tau': c_params.mirostat_tau,
            'mirostat_eta': c_params.mirostat_eta,
            'skip_special_token': bool(c_params.skip_special_token),
            'is_async': bool(c_params.is_async),
            'img_start': "".encode('utf-8'),
            'img_end': "".encode('utf-8'),
            'img_content': "".encode('utf-8'),
            'extend_param': {
                'base_domain_id': c_params.extend_param.base_domain_id,
                'embed_flash': c_params.extend_param.embed_flash,
                'enabled_cpus_num': c_params.extend_param.enabled_cpus_num,
                'enabled_cpus_mask': c_params.extend_param.enabled_cpus_mask
            }
        }

    def init(self, params: Optional[Dict[str, Any]] = None, 
             callback: Optional[Callable[[str, int, Dict], None]] = None) -> None:
        if params is None:
            raise RuntimeError(f"RKLLM initialization required params")
        if 'model_path' not in params:
            raise RuntimeError(f"RKLLM initialization required model_path in params")

        # Convert params to ctypes structure
        c_params = self._lib.rkllm_createDefaultParam()
        c_params.model_path = params.get('model_path').encode('utf-8')
        c_params.max_context_len = params.get('max_context_len', 4096)
        c_params.max_new_tokens = params.get('max_new_tokens', 2048)
        c_params.top_k = params.get('top_k', 1)
        c_params.top_p = params.get('top_p', 0.8)
        c_params.MinP = params.get('MinP', 0)
        c_params.skip_special_token = True
        c_params.img_start = params.get('img_start', "".encode('utf-8'))
        c_params.img_end = params.get('img_end', "".encode('utf-8'))
        c_params.img_content = params.get('img_content', "".encode('utf-8'))
        extend_param = params.get('extend_param', None)
        if extend_param is not None:
            c_params.extend_param.base_domain_id = extend_param.get('base_domain_id', 0)
            c_params.extend_param.enabled_cpus_num = extend_param.get('enabled_cpus_num', 4)
            c_params.extend_param.enabled_cpus_mask = extend_param.get('enabled_cpus_mask', CPU4|CPU5|CPU6|CPU7)
        
        # Setup callback if provided
        if callback:
            @staticmethod
            def wrapped_callback(result_ptr, userdata, state):
                if state == State.FINISH:
                    callback("\n", 0, state)
                elif state == State.ERROR:
                    callback("<error>\n", 0, state)
                else:
                    result = result_ptr.contents
                    text = result.text.decode('utf-8') if result.text else ""
                    callback(text, result.token_id, state)
            self._current_callback = _LLMResultCallback(wrapped_callback)
        else:
            self._current_callback = None
        
        # Call init
        ret = self._lib.rkllm_init(ctypes.byref(self._handle), ctypes.byref(c_params), 
                                  self._current_callback)
        if ret != 0:
            raise RuntimeError(f"RKLLM initialization failed with error code {ret}")

    def load_lora(self, adapter_path: str, adapter_name: str, scale: float = 1.0) -> None:
        adapter = self._RKLLMLoraAdapter()
        adapter.lora_adapter_path = adapter_path.encode('utf-8')
        adapter.lora_adapter_name = adapter_name.encode('utf-8')
        adapter.scale = scale
        
        ret = self._lib.rkllm_load_lora(self._handle, ctypes.byref(adapter))
        if ret != 0:
            raise RuntimeError(f"Failed to load LoRA adapter with error code {ret}")

    def load_prompt_cache(self, cache_path: str) -> None:
        ret = self._lib.rkllm_load_prompt_cache(self._handle, cache_path.encode('utf-8'))
        if ret != 0:
            raise RuntimeError(f"Failed to load prompt cache with error code {ret}")

    def release_prompt_cache(self) -> None:
        ret = self._lib.rkllm_release_prompt_cache(self._handle)
        if ret != 0:
            raise RuntimeError(f"Failed to release prompt cache with error code {ret}")

    def destroy(self) -> None:
        if hasattr(self, '_handle') and self._handle:
            try:
                ret = self._lib.rkllm_destroy(self._handle)
                if ret != 0:
                    print(f"Warning: RKLLM destruction returned error code {ret}")
                self._handle = None
                self._current_callback = None
            except Exception as e:
                print(f"Error during destruction: {str(e)}")

    def run(self, input_data: str| RKLLMMultiModelInput, 
            infer_params: RKLLMInferParam) -> None:
        c_input = _RKLLMInput()
        
        if isinstance(input_data, str):
            # Text prompt input
            c_input.input_mode = InputType.PROMPT
            c_input.input_data.prompt_input  = input_data.encode('utf-8')
        elif isinstance(input_data, RKLLMMultiModelInput):
            # Token IDs input
            c_input.input_mode = InputType.MULTIMODAL
            c_input.input_data.multimodal_input.prompt = input_data.prompt.encode('utf-8')
            c_input.input_data.multimodal_input.image_embed = input_data.image_embed.ctypes.data_as(POINTER(c_float))
            c_input.input_data.multimodal_input.n_image_tokens = ctypes.c_size_t(input_data.n_image_tokens)
            c_input.input_data.multimodal_input.n_image = ctypes.c_size_t(input_data.n_image)
            c_input.input_data.multimodal_input.image_width = ctypes.c_size_t(input_data.image_width)
            c_input.input_data.multimodal_input.image_height = ctypes.c_size_t(input_data.image_height)
        else:
            raise ValueError("Input must be string, RKLLMMultiModelInput")
        
        # Prepare inference params
        c_infer_params = _RKLLMInferParam()
        c_infer_params.mode = infer_params.mode
        if infer_params.lora_params:
            c_infer_params.lora_params.lora_adapter_name = infer_params.lora_params.lora_adapter_name
        if infer_params.prompt_cache_params:
            c_infer_params.prompt_cache_params.save_prompt_cache = infer_params.prompt_cache_params.save_prompt_cache
            c_infer_params.prompt_cache_params.prompt_cache_path = infer_params.prompt_cache_params.prompt_cache_path
        c_infer_params.keep_history = infer_params.keep_history
        
        ret = self._lib.rkllm_run(self._handle, ctypes.byref(c_input), 
                                ctypes.byref(c_infer_params), None)
        if ret != 0:
            raise RuntimeError(f"Inference failed with error code {ret}")

    def run_async(self, input_data: str| RKLLMMultiModelInput, 
            infer_params: RKLLMInferParam) -> None:
        c_input = _RKLLMInput()
        
        if isinstance(input_data, str):
            # Text prompt input
            c_input.input_mode = InputType.PROMPT
            c_input.input_data.prompt_input  = input_data.encode('utf-8')
        elif isinstance(input_data, RKLLMMultiModelInput):
            # Token IDs input
            c_input.input_mode = InputType.MULTIMODAL
            c_input.input_data.multimodal_input.prompt = ctypes.c_char_p(input_data.prompt.encode('utf-8'))
            c_input.input_data.multimodal_input.image_embed = ctypes.cast(input_data.image_embed, POINTER(c_float))
            c_input.input_data.multimodal_input.n_image_tokens = ctypes.c_size_t(input_data.n_image_tokens)
            c_input.input_data.multimodal_input.n_image = ctypes.c_size_t(input_data.n_image)
            c_input.input_data.multimodal_input.image_width = ctypes.c_size_t(input_data.image_width)
            c_input.input_data.multimodal_input.image_height = ctypes.c_size_t(input_data.image_height)
        else:
            raise ValueError("Input must be string, RKLLMMultiModelInput")
        
        # Prepare inference params
        c_infer_params = _RKLLMInferParam()
        c_infer_params.mode = infer_params.mode
        c_infer_params.lora_params.lora_adapter_name = infer_params.lora_params.lora_adapter_name
        c_infer_params.prompt_cache_params.save_prompt_cache = infer_params.prompt_cache_params.save_prompt_cache
        c_infer_params.prompt_cache_params.prompt_cache_path = infer_params.prompt_cache_params.prompt_cache_path
        c_infer_params.keep_history = infer_params.keep_history
        
        ret = self._lib.rkllm_run_async(self._handle, ctypes.byref(c_input), 
                                      ctypes.byref(c_infer_params), None)
        if ret != 0:
            raise RuntimeError(f"Async inference failed with error code {ret}")

    def abort(self) -> None:
        ret = self._lib.rkllm_abort(self._handle)
        if ret != 0:
            raise RuntimeError(f"Failed to abort with error code {ret}")

    def is_running(self) -> bool:
        return bool(self._lib.rkllm_is_running(self._handle))

    def clear_kv_cache(self, keep_system_prompt: bool = False) -> None:
        ret = self._lib.rkllm_clear_kv_cache(self._handle, keep_system_prompt)
        if ret != 0:
            raise RuntimeError(f"Failed to load prompt cache with error code {ret}")

    def rkllm_set_chat_template(self, system_prompt, prompt_prefix, prompt_postfix) -> None:
        ret =  self._lib.rkllm_set_chat_template(
        self._handle,
        system_prompt.encode('utf-8') if system_prompt else None,
        prompt_prefix.encode('utf-8') if prompt_prefix else None,
        prompt_postfix.encode('utf-8') if prompt_postfix else None
        )
        if ret != 0:
            raise RuntimeError(f"Failed to load prompt cache with error code {ret}")

