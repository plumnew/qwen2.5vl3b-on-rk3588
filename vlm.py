import os
import sys
import time
import threading
import argparse
import json
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from rkllm_adapter import *
#from image_enc_rknnlite import RKNNImageEncoder
from image_enc_rknnlite import RKNNImageEncoder

app = Flask(__name__)
CORS(app)
# Create a lock to control multi-user access to the server
lock = threading.Lock()

# Global variables for callback state
global_text = ''
global_state = -1
is_blocking = False

def expand_and_resize(img, background_color=(127.5, 127.5, 127.5), target_size=(392, 392)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]
    if width == height:
        expanded_img = img.copy()
    else:
        size = max(width, height)
        expanded_img = np.full((size, size, 3), background_color, dtype=img.dtype)
        x_offset = (size - width) // 2
        y_offset = (size - height) // 2
        expanded_img[y_offset:y_offset+height, x_offset:x_offset+width] = img

    resized_img = cv2.resize(expanded_img, target_size)

    return resized_img

class VLMManager:
    def __init__(self, rkllm_model_path, rknn_model_path):
        # Initialize RKLLM model
        print("=========init RKLLM....===========")
        sys.stdout.flush()
        
        self.rkllm = RKLLM()
        params = {
            'model_path': rkllm_model_path,
            'max_context_len': 4096,
            'max_new_tokens': 2048,
            'top_k': 1,
            'img_start' : '<|vision_start|>'.encode('utf-8'),
            'img_end' : '<|vision_end|>'.encode('utf-8'),
            'img_content' : '<|image_pad|>'.encode('utf-8'),
            'extend_param':{
                    'base_domain_id': 1,
                    'enabled_cpus_num': 4,
                    'enabled_cpus_mask': CPU4|CPU5|CPU6|CPU7
                }
        }
        self.rkllm.init(params, VLMManager.generation_callback)
        
        # Set chat template
        system_prompt = "<|im_start|>system\n你的任务是描述图片中的内容<|im_end|>\n"
        prompt_prefix = "<|im_start|>user\n"
        prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"
        self.rkllm.rkllm_set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
        
        print("RKLLM Model has been initialized successfully!")
        print("==============================")
        sys.stdout.flush()

        # Initialize RKNN image encoder
        print("=========init RKNN image encoder....===========")
        sys.stdout.flush()
        self.img_encoder = RKNNImageEncoder()
        self.img_encoder.init_imgenc(rknn_model_path)
        print("RKNN image encoder has been initialized successfully!")
        print("==============================")
    
    @staticmethod
    def generation_callback(text: str, token_id: int, state: int):
        global global_text
        global_text+=text
        print(text, end='', flush=True)

    def process_image(self, base64_image):
        # Decode base64 image
        img_data = base64.b64decode(base64_image.split(",")[1])
        nparr = np.frombuffer(img_data, dtype=np.uint8)
        mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = expand_and_resize(mat)
        img_embed = self.img_encoder.run_imgenc(img)
        return img_embed

    def generate_response(self, prompt, img_embed=None):
        infer_params = RKLLMInferParam()
        infer_params.mode = InferMode.GENERATE
        infer_params.keep_history = 0
        infer_params.lora_params = None
        infer_params.prompt_cache_params = None

        if img_embed is not None:
            # Multimodal input
            multi_input = RKLLMMultiModelInput()
            multi_input.prompt = prompt
            multi_input.image_embed = img_embed
            multi_input.n_image_tokens = 196
            multi_input.n_image = 1
            multi_input.image_width = 392
            multi_input.image_height = 392
            
            self.rkllm.run(multi_input, infer_params)
        else:
            self.rkllm.run(prompt, infer_params)

    def release(self):
        if hasattr(self, 'img_encoder'):
            self.img_encoder.release_imgenc()
        if hasattr(self, 'rkllm'):
            self.rkllm.destroy()

@app.route('/rkllm_chat', methods=['POST'])
def receive_message():
    global global_text, global_state, is_blocking

    if is_blocking or global_state == 0:
        return jsonify({'status': 'error', 'message': 'RKLLM_Server is busy! Try again later.'}), 503
    
    lock.acquire()
    try:
        is_blocking = True
        data = request.json
        if not data or 'messages' not in data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON data!'}), 400

        global_text = ''
        global_state = -1

        rkllm_responses = {
            "id": "rkllm_chat",
            "object": "rkllm_chat",
            "created": None,
            "choices": [],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }

        messages = data['messages']

        prompt_texts = []
        img_embed = None

        for message in messages:
            if 'content' not in message:
                continue

            content = message['content']

            if isinstance(content, dict) and content.get("type") == "imagedata":
                imagedata = content.get("imagedata")
                if imagedata.startswith("data:image/jpeg;base64"):
                    print("图片转换为Token")
                    start_time = time.time()
                    img_embed = vlm_manager.process_image(imagedata)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"调用时间: {elapsed_time:.6f} 秒，开始推理...")
                    prompt_texts.append("<image>")
            elif isinstance(content, str):
                prompt_texts.append(content)

        full_prompt = "".join(prompt_texts)

        def generate():
            vlm_manager.generate_response(full_prompt, img_embed)
            return global_text

        rkllm_output = generate()

        rkllm_responses["choices"].append({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": rkllm_output,
            },
            "logprobs": None,
            "finish_reason": "stop"
        })

        return jsonify(rkllm_responses), 200

            
    finally:
        lock.release()
        is_blocking = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, 
                       help='Absolute path of the converted RKLLM model')
    parser.add_argument('--rknn_model_path', type=str, required=True,
                       help='Absolute path of the converted RKNN model')
    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Invalid rkllm model path")
        sys.exit(1)

    if not os.path.exists(args.rknn_model_path):
        print("Error: Invalid img_model path")
        sys.exit(1)

    # Initialize VLM manager
    vlm_manager = VLMManager(args.rkllm_model_path, args.rknn_model_path)

    try:
        app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
    finally:
        vlm_manager.release()
        print("RKLLM resources released")
