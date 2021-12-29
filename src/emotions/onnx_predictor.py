import numpy as np
import onnxruntime


class OnnxPredictor:

    def __init__(self, labels, model_path=f'../resources/models/emotions/emotions.onnx'):
        self.labels = labels
        self.session = onnxruntime.InferenceSession(model_path, None)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, data):
        data = data.astype(np.float32)
        pred_onnx = self.session.run([self.output_name], {self.input_name: data})
        max_index = int(np.argmax(np.squeeze(pred_onnx)))
        print(f"OnnxPredictor: [{self.labels[max_index]}]")
        return self.labels[max_index]

