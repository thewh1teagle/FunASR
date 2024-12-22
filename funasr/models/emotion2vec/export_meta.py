#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch

def export_rebuild_model(model, **kwargs):
    """Rebuild model for ONNX export by adding necessary export methods."""
    model.device = kwargs.get("device", "cpu")
    # model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)
    return model

def export_dummy_inputs(self):
    """Create dummy inputs for ONNX export."""
    # Create a dummy audio input of 1 second (16000 samples) for batch size 1
    return (torch.randn(1, 16000 * 1),)

def export_input_names(self):
    """Define input names for ONNX export."""
    return ["source"]

def export_output_names(self):
    """Define output names for ONNX export."""
    if self.proj is not None:
        return ["logits"]
    return ["embeddings"]

def export_dynamic_axes(self):
    """Define dynamic axes for ONNX export."""
    # return None
    return {
        "source": {0: "batch_size", 1: "sequence_length"},
        "embeddings": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
    
    

def export_name(self):
    """Define ONNX model name."""
    return "emotion2vec.onnx"