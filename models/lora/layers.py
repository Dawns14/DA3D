import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self
import spconv.pytorch as spconv
import threading
from contextlib import contextmanager
import inspect

class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):

        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.merged = False

class LoRALinear(LoRALayer):

    def __init__(
        self,
        linear_layer: nn.Linear,

        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        tasks=None,
        **kwargs,
    ):

        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = linear_layer
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        if r > 0:
            self.lora_A = nn.Parameter(
                self.linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r

            self.reset_parameters()

    def __repr__(self):
        s = super().__repr__()
        if self.r > 0:
            s += f"\n  (lora_A): {self.lora_A.shape}"
            s += f"\n  (lora_B): {self.lora_B.shape}"
        return s

    def reset_parameters(self):

        if hasattr(self, "lora_A"):

            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):

        if self.r > 0 and not self.merged:

            self.linear.weight.data += (self.lora_B @
                                        self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):

        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)) * self.scaling
        return pretrained + lora

class LoRAConv(LoRALayer):
    def __init__(
        self,
        conv_layer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.conv = conv_layer
        self.scaling = self.lora_alpha

        if r > 0:
            kernel_size = self.conv.kernel_size
            in_channels = self.conv.in_channels
            out_channels = self.conv.out_channels
            groups = self.conv.groups

            A_kernel_size = 1
            for ks in range(len(kernel_size)-1):
                A_kernel_size *= kernel_size[ks]
            B_kernel_size = kernel_size[-1]

            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r, in_channels * A_kernel_size))
            )

            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_channels // groups * B_kernel_size, r))
            )

        self.reset_parameters()
        self.merged = False

    def __repr__(self):
        s = super().__repr__()
        if self.r > 0:
            s += f"\n  (lora_A): {self.lora_A.shape}"
            s += f"\n  (lora_B): {self.lora_B.shape}"
        return s

    def reset_parameters(self):

        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):

        if self.r > 0 and not self.merged:

            delta_weight = (self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling

            kernel_size = self.conv.kernel_size
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[0]
            delta_weight = delta_weight.view(
                self.conv.out_channels // self.conv.groups,
                self.conv.in_channels,
                kernel_size,
                kernel_size,
            )

            self.conv.weight.data += delta_weight
            self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            try:
                return self.conv._conv_forward(
                    input = x,
                    weight = self.conv.weight + (self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)).view(self.conv.weight.shape) * self.scaling,
                    bias = self.conv.bias,
                    training = self.conv.training
                )
            except TypeError:
                return self.conv._conv_forward(
                    input = x,
                    weight = self.conv.weight + (self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)).view(self.conv.weight.shape) * self.scaling,
                    bias = self.conv.bias
                )
        return self.conv(x)

class LoRASparseConv(LoRAConv, spconv.SparseModule):
    def __init__(
        self,
        conv_layer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):

        LoRAConv.__init__(self, conv_layer, r, lora_alpha, lora_dropout)

class MTLoRABase:
    _local = threading.local()

    @classmethod
    def get_current_weather(cls):
        if not hasattr(cls._local, 'weather_stack') or not cls._local.weather_stack:
            raise RuntimeError("No active weather context. Use MTLoRABase.weather_context()")
        return cls._local.weather_stack[-1]

    @classmethod
    @contextmanager
    def weather_context(cls, weather: str):

        if not hasattr(cls._local, 'weather_stack'):
            cls._local.weather_stack = []

        cls._local.weather_stack.append(weather)

        try:
            yield
        finally:
            if cls._local.weather_stack:
                cls._local.weather_stack.pop()

    @classmethod
    def validate_weather(cls, weather_list):
        current = cls.get_current_weather()
        if current not in weather_list:
            raise ValueError(f"Current weather '{current}' not in layer's weather list: {weather_list}")

class MTLoRAConv(LoRALayer, MTLoRABase):
    def __init__(
        self,
        conv_layer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        AdaLoRA: bool = False,
    ):
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.conv = conv_layer
        self.scaling = self.lora_alpha
        self.weather_list= ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']
        self.intermediate_products = {}
        self.intermediate_products_count = {}
        self.intermediate_grad = {}
        self.intermediate_grad_count = {}
        self.rank = {weather: r for weather in self.weather_list}
        self.r = r
        self.AdaLoRA = AdaLoRA

        if r > 0:
            kernel_size = self.conv.kernel_size
            in_channels = self.conv.in_channels
            out_channels = self.conv.out_channels
            groups = self.conv.groups

            A_kernel_size = 1
            for ks in range(len(kernel_size)-1):
                A_kernel_size *= kernel_size[ks]
            B_kernel_size = kernel_size[-1]
            min_r = min(in_channels * A_kernel_size, out_channels * B_kernel_size)
            self.r = min(self.r, min_r)
            self.rank = {weather: self.r for weather in self.weather_list}

            self.lora_A = nn.ParameterDict({
                weather: nn.Parameter(
            self.conv.weight.new_zeros((self.r*3 if self.AdaLoRA else self.r, in_channels * A_kernel_size))
                ) for weather in self.weather_list
            })

            self.lora_B = nn.ParameterDict({
                weather: nn.Parameter(
                    self.conv.weight.new_zeros((out_channels // groups * B_kernel_size, self.r*3 if self.AdaLoRA else self.r))
                ) for weather in self.weather_list
            })

        self.reset_parameters()

    def __repr__(self):
        s = super().__repr__()
        if self.r > 0:
            s += f"\n  AdaLoRA: {self.AdaLoRA}"
            s += f"\n  Current ranks: {self.rank}"
            for weather in self.weather_list:
                if self.AdaLoRA:

                    actual_A_shape = (self.rank[weather], self.lora_A[weather].shape[1])
                    actual_B_shape = (self.lora_B[weather].shape[0], self.rank[weather])
                    s += f"\n  (lora_A[{weather}]): {actual_A_shape} (allocated: {self.lora_A[weather].shape})"
                    s += f"\n  (lora_B[{weather}]): {actual_B_shape} (allocated: {self.lora_B[weather].shape})"
                else:
                    s += f"\n  (lora_A[{weather}]): {self.lora_A[weather].shape}"
                    s += f"\n  (lora_B[{weather}]): {self.lora_B[weather].shape}"
        return s

    def get_actual_lora_params(self):

        if self.r == 0:
            return 0

        total = 0
        for weather in self.weather_list:
            if self.AdaLoRA:

                in_features = self.lora_A[weather].shape[1]
                out_features = self.lora_B[weather].shape[0]
                total += self.rank[weather] * in_features
                total += out_features * self.rank[weather]
            else:

                total += self.lora_A[weather].numel()
                total += self.lora_B[weather].numel()
        return total

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            for weather in self.weather_list:
                nn.init.kaiming_uniform_(self.lora_A[weather], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[weather])

    def adaptive_lora_rank(self, k=4):
        with torch.no_grad():
            thetas = {}
            grad = {}
            pruned_A = {}
            pruned_B = {}
            need_break = False
            for weather in self.weather_list:
                A = self.lora_A[weather][:self.rank[weather], :].clone()
                B = self.lora_B[weather][:,:self.rank[weather]].clone()

                C = B @ A
                U, S, Vh = torch.linalg.svd(C, full_matrices=True)

                assert self.r <= S.shape[0]

                if self.r == S.shape[0]:
                    return

                S_pruned = torch.zeros(U.shape[0], Vh.shape[0], device=C.device)
                if S.shape[0] > self.rank[weather]-k:
                    S_pruned[:self.rank[weather]-k, :self.rank[weather]-k] = torch.diag(S[:self.rank[weather]-k])
                    C_prime = (U @ S_pruned) @ Vh
                    err = C - C_prime
                    pruned_B[weather] = U[:, :self.rank[weather]-k] @ S_pruned[:self.rank[weather]-k, :self.rank[weather]-k]
                    pruned_A[weather] = Vh[:self.rank[weather]-k, :]
                else:
                    S_pruned[:S.shape[0], :S.shape[0]] = torch.diag(S)
                    err = torch.zeros_like(C)
                    need_break = True

                grad_C = self.intermediate_grad[weather] / self.intermediate_grad_count[weather]
                thetas[weather] = torch.norm(torch.mul(err, grad_C))
                grad[weather] = torch.norm(grad_C)
                self.intermediate_grad[weather] = torch.zeros_like(grad_C)
                self.intermediate_grad_count[weather] = 0

            if need_break:

                return

            sorted_thetas = dict(sorted(thetas.items(), key=lambda x: x[1]))
            prune_rank = 0
            for i in range(2):
                weather = list(sorted_thetas.keys())[i]
                if self.rank[weather] <= self.r // 4:
                    continue
                nn.init.zeros_(self.lora_B[weather][:, self.rank[weather]-k:self.rank[weather]])
                nn.init.kaiming_uniform_(self.lora_A[weather][self.rank[weather]-k:self.rank[weather], :], a=math.sqrt(5))
                self.lora_B[weather][:, :self.rank[weather]-k] = pruned_B[weather]
                self.lora_A[weather][:self.rank[weather]-k, :] = pruned_A[weather]
                self.rank[weather] -= k
                prune_rank += k

            sorted_grad = dict(sorted(grad.items(), key=lambda x: x[1]))
            for i in range(prune_rank // k):
                weather = list(sorted_grad.keys())[-1-i]
                self.rank[weather] += k

    def current_rank(self):
        return self.rank

    def calculate_grad(self, weather):

        with torch.no_grad():
            if weather not in self.intermediate_products:
                return
            product = self.intermediate_products[weather]
            assert len(product) == 1
            grad = product[0].grad.detach().clone()
            if weather not in self.intermediate_grad:
                self.intermediate_grad[weather] = grad
                self.intermediate_grad_count[weather] = 1
            else:
                self.intermediate_grad[weather] += grad
                self.intermediate_grad_count[weather] += 1
            self.intermediate_products[weather] = []

    def forward(self, x):
        if self.r > 0:
            weather = self.get_current_weather()
            self.validate_weather(self.weather_list)
            if self.AdaLoRA:
                product = self.lora_B[weather][:,:self.rank[weather]] @ self.lora_A[weather][:self.rank[weather], :]
                if self.training:
                    product.retain_grad()
                    if weather not in self.intermediate_products:
                        self.intermediate_products[weather] = [product]
                    else:
                        self.intermediate_products[weather].append(product)
            else:
                product = self.lora_B[weather] @ self.lora_A[weather]
            try:
                return self.conv._conv_forward(
                    input = x,
                    weight = self.conv.weight + product.view(self.conv.weight.shape) * self.scaling,
                    bias = self.conv.bias,
                    training = self.conv.training
                )
            except TypeError:
                return self.conv._conv_forward(
                    input = x,
                    weight = self.conv.weight + product.view(self.conv.weight.shape) * self.scaling,
                    bias = self.conv.bias
                )
        return self.conv(x)

class MTLoRASparseConv(MTLoRAConv, spconv.SparseModule):
    def __init__(
        self,
        conv_layer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        AdaLoRA: bool = False,
    ):

        MTLoRAConv.__init__(self, conv_layer, r, lora_alpha, lora_dropout,  AdaLoRA)

class MTBatchNorm(nn.Module, MTLoRABase):
    def __init__(self, batchnorm_layer):
        super().__init__()

        self.weather_list = [
            'normal', 'overcast', 'fog',
            'rain', 'sleet', 'lightsnow', 'heavysnow'
        ]

        self.batchnorms = nn.ModuleDict({
            weather: self._clone_bn(batchnorm_layer)
            for weather in self.weather_list
        })

    def _clone_bn(self, original_bn):

        new_bn = type(original_bn)(original_bn.num_features)

        new_bn.load_state_dict(original_bn.state_dict())

        new_bn.train(mode=original_bn.training)
        new_bn.cuda()
        return new_bn

    def forward(self, x):
        weather = self.get_current_weather()
        self.validate_weather(weather)
        return self.batchnorms[weather](x)
