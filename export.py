"""
uv venv
uv pip install -r requirements.txt
uv run export.py
"""
from funasr import AutoModel

model = AutoModel(
    model="iic/emotion2vec_plus_large"
)

res = model.export(type="onnx", quantize=False, opset_version=13, device='cpu', disable_update=True, output_dir='.')  # fp32 onnx-cpu
