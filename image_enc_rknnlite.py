from rknnlite.api import RKNNLite
import numpy as np

class RKNNImageEncoder:
    def __init__(self):
        self.rknn_ctx = None
        self.model_channel = 0
        self.model_width = 0
        self.model_height = 0

    def init_imgenc(self, model_path):
        # Load RKNN Model
        self.rknn_ctx = RKNNLite()
        
        # Load model from file
        ret = self.rknn_ctx.load_rknn(model_path)
        if ret != 0:
            print("load_rknn fail!")
            return -1
        
        # Init runtime environment
        ret = self.rknn_ctx.init_runtime()
        if ret != 0:
            print("init_runtime fail!")
            return -1
        
        return 0

    def release_imgenc(self):
        if self.rknn_ctx is not None:
            self.rknn_ctx.release()
            self.rknn_ctx = None
        return 0

    def run_imgenc(self, img_data):
        # Prepare input as a list of numpy arrays
        inputs = [np.expand_dims(img_data, axis=0)]
        
        # Run inference
        outputs = self.rknn_ctx.inference(inputs=inputs, data_format='nhwc')
        if outputs is None:
            print("inference fail!")
            return None
        
        # The output is already a numpy array, no need for further conversion
        out_result = outputs[0]
        return np.ascontiguousarray(out_result, dtype=np.float32)