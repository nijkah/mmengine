# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import MODEL_WRAPPERS

try:
    from deepspeed.runtime.engine import DeepSpeedEngine
    MODEL_WRAPPERS.register_module(module=DeepSpeedEngine)
except ImportError:
    DeepSpeedEngine = None
