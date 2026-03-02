import torch.nn as nn

from models import pre_processor, backbone_3d, head, roi_head, img_cls, lora
import contextlib
from collections import Counter

class RL3DF_gate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.MODEL

        self.is_multi_head = cfg.MODEL.get("MULTI_HEAD", {}).get("IS_MULTI_HEAD", False)
        self.is_mt_lora = cfg.MODEL.get("LoRA", {}).get("MTLoRA", False)
        self.weather_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

        self.list_module_names = [
            'pre_processor', 'pre_processor2', 'img_cls', 'backbone_3d', 'head', 'roi_head',
        ]
        self.list_modules = []
        self.build_rl_detector()

    def build_rl_detector(self):
        for name_module in self.list_module_names:
            if self.is_multi_head and (name_module == "head" or name_module == "roi_head"):
                for weather in self.weather_list:
                    module = getattr(self, f"build_{name_module}")()
                    if module is not None:
                        self.add_module(f"{name_module}_{weather}", module)

            else:
                module = getattr(self, f"build_{name_module}")()
                if module is not None:
                    self.add_module(name_module, module)
                    self.list_modules.append(module)

    def build_img_cls(self):
        if self.cfg_model.get('IMG_CLS', None) is None:
            return None

        module = img_cls.__all__[self.cfg_model.IMG_CLS.NAME]()
        return module

    def build_pre_processor(self):
        if self.cfg_model.get('PRE_PROCESSOR', None) is None:
            return None

        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR.NAME](self.cfg)
        return module

    def build_pre_processor2(self):
        if self.cfg_model.get('PRE_PROCESSOR2', None) is None:
            return None

        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR2.NAME](self.cfg)
        return module

    def build_backbone_3d(self):
        cfg_backbone = self.cfg_model.get('BACKBONE', None)
        return backbone_3d.__all__[cfg_backbone.NAME](self.cfg)

    def build_head(self):
        if (self.cfg.MODEL.get('HEAD', None)) is None:
            return None
        module = head.__all__[self.cfg_model.HEAD.NAME](self.cfg)
        return module

    def build_roi_head(self):
        if (self.cfg.MODEL.get('ROI_HEAD', None)) is None:
            return None
        head_module = roi_head.__all__[self.cfg_model.ROI_HEAD.NAME](self.cfg)
        return head_module

    def _get_context(self, weather):
        if self.is_mt_lora:
            return lora.MTLoRABase.weather_context(weather)
        return contextlib.nullcontext()

    def _resolve_weather(self, x):
        weather = x.get('weather_context', None)
        if weather is not None:
            return weather

        pred_weather_list = x.get('pred_weather_list', None)
        if pred_weather_list is not None and len(pred_weather_list) > 0:
            weather = Counter(pred_weather_list).most_common(1)[0][0]
            x['weather_context'] = weather
            return weather

        weather_conditions = [sample["desc"]["climate"] for sample in x["meta"]]
        assert len(set(weather_conditions)) == 1
        weather = weather_conditions[0]
        x['weather_context'] = weather
        return weather

    def forward(self, x):
        if self.is_multi_head or self.is_mt_lora:
            weather = self._resolve_weather(x)
            context = self._get_context(weather)
            with context:
                for module in self.list_module_names:
                    if self.is_multi_head and (module == "head" or module == "roi_head"):
                        head_module = getattr(self, f"{module}_{weather}", None)
                        if head_module is not None:
                            x = head_module(x)
                    else:
                        cur_module = getattr(self, module, None)
                        if cur_module is not None:
                            x = cur_module(x)
        else:
            for module in self.list_modules:
                x = module(x)
        return x
