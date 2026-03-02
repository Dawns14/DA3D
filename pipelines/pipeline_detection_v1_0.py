import torch
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import shutil
from torch.utils.data import Subset

from numba.core.errors import NumbaWarning
import warnings
import logging
import sys
import copy
from models.lora import *
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from torch import nn
from models.lora import MTLoRABase, MTLoRAConv, MTLoRASparseConv, LoRASparseConv, LoRAConv
from models.img_cls.cls_model_resnet import ImageClsBackbone as WeatherClsResNet18

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

warnings.simplefilter('ignore', category=NumbaWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)

from torch.utils.tensorboard import SummaryWriter

from utils.util_pipeline import *
from utils.util_point_cloud import *
from utils.util_config import cfg, cfg_from_yaml_file

from utils.util_point_cloud import Object3D
import utils.kitti_eval.kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result
from utils.kitti_eval.eval_revised import get_official_eval_result_revised

from utils.util_optim import clip_grad_norm_

class PipelineDetection_v1_0():
    def __init__(self, path_cfg=None, mode='train'):

        self.cfg = cfg_from_yaml_file(path_cfg, cfg)
        self.mode = mode
        self.update_cfg_regarding_mode()

        self.weather_list= ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']
        if self.cfg.GENERAL.SEED is not None:
            try:
                set_random_seed(cfg.GENERAL.SEED, cfg.GENERAL.IS_CUDA_SEED, cfg.GENERAL.IS_DETERMINISTIC)
            except:
                print('* Exception error: check cfg.GENERAL for seed')
                set_random_seed(cfg.GENERAL.SEED)

        print('* K-Radar dataset is being loaded.')
        self.dataset_train = build_dataset(self, split='train') if self.mode == 'train' else None
        self.dataset_test = build_dataset(self, split='test')
        print('* The dataset is loaded.')
        if mode == 'train':
            self.cfg.DATASET.NUM = len(self.dataset_train)
        elif mode in ['test', 'vis']:
            self.cfg.DATASET.NUM = len(self.dataset_test)

        self.head_names = self.cfg.MODEL.get('head_names',  ['head'])
        self.network = build_network(self).cuda()
        self.optimizer = build_optimizer(self, self.network)
        self.scheduler = build_scheduler(self, self.optimizer)
        self.epoch_start = 0
        self.open_lora = False
        self.open_multi_head = False
        self.mtoptim = False
        self.best_weather_results = {}
        self.best_weather_models = {}
        self.weather_classifier = None
        self.use_weather_classifier = False

        if self.cfg.MODEL.get('MULTI_HEAD', {}).get('IS_MULTI_HEAD', False):
            self.open_multi_head = True

        if self.cfg.GENERAL.LOGGING.IS_LOGGING:
            self.set_logging(path_cfg)

        if self.cfg.VAL.IS_VALIDATE:
            self.set_validate()
        else:
            self.is_validate = False

        if self.cfg.GENERAL.RESUME.IS_RESUME and self.cfg.GENERAL.RESUME.get('LOAD_BEST_IOU', None) is None:
            self.resume_network()

        if self.cfg.MODEL.get('LoRA', None) is not None:
            self.open_lora = True
            self.ada_lora = self.cfg.MODEL.LoRA.get('AdaLoRA', False)
            self.ada_end_epoch = self.cfg.MODEL.LoRA.get('AdaLoRA_end_epoch', 20)
            self.ada_epoch_freq = self.cfg.MODEL.LoRA.get('AdaLoRA_freq', 1)
            ignore_module = self.cfg.MODEL.LoRA.get('ignore', [])
            for name, module in self.network.named_children():
                if name in self.head_names or ('_' in name and name.rsplit('_', 1)[0] in self.head_names):
                    continue
                if name in ignore_module:
                    continue
                apply_lora(self, module)
            if self.cfg.MODEL.LoRA.get('only_lora', True):
                mark_only_lora_as_trainable(self.network,
                                            head=self.cfg.MODEL.LoRA.get('head_train', True),
                                            batchnorm=self.cfg.MODEL.LoRA.get('batchnorm_train', True))
            self.optimizer = build_optimizer(self, self.network)
            self.scheduler = build_scheduler(self, self.optimizer)
            print('* LoRA is applied. Reset optimizer and scheduler.')

        if self.cfg.OPTIMIZER.get('MTOPTIM', False):
            self.mtoptim = True
            self.optimizer, self.scheduler = build_optimizer_mt(self, self.network, self.dataset_train)
            # print('* MT-Optimizer is applied. Reset optimizer and scheduler.')

        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)
        self.get_loss_from = self.cfg.get('get_loss_from', 'head')
        self.optim_fastai = True\
            if self.cfg.OPTIMIZER.NAME in ['adam_onecycle', 'adam_cosineanneal'] else False
        self.grad_norm_clip = self.cfg.OPTIMIZER.get('GRAD_NORM_CLIP', -1)

        self.set_vis()
        self.set_weather_classifier()


        self.is_validation_updated = self.cfg.get('is_validation_updated', False)
        self.is_validate_best = self.cfg.get('is_validate_best', False)

        cfg_distil = self.cfg.get('DISTIL', None)
        if cfg_distil is not None:
            self.distil = True
            self.infer_head_of_distil_model = cfg_distil.get('INFER_HEAD', False)
            import yaml
            from easydict import EasyDict
            with open(cfg_distil.CFG, 'r') as f:
                new_config = yaml.safe_load(f)
            from models.skeletons import build_skeleton
            distil_model = build_skeleton(EasyDict(new_config))

            if not self.infer_head_of_distil_model:
                if hasattr(distil_model, 'head'):
                    import torch.nn as nn
                    distil_model.head = nn.Identity()
            distil_model.load_state_dict(torch.load(cfg_distil.PTH), strict=False)
            self.distil_model = distil_model.cuda().eval()

            print('* The model for distilation is loaded.')
        else:
            self.distil = False

    def update_cfg_regarding_mode(self):

        if self.mode == 'train':
            pass
        elif self.mode == 'test':
            self.cfg.OPTIMIZER.NUM_WORKERS = 0
        elif self.mode == 'vis':
            self.cfg.OPTIMIZER.NUM_WORKERS = 0
            self.cfg.GET_ITEM = {
                'rdr_sparse_cube'   : True,
                'rdr_tesseract'     : False,
                'rdr_cube'          : True,
                'rdr_cube_doppler'  : False,
                'ldr_pc_64'         : True,
                'cam_front_img'     : True,
            }
        else:
            print('* Exception error (Pipeline): check modify_cfg')
        return

    def set_validate(self):
        self.is_validate = True
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD
        list_val_keyword_keys = list(self.val_keyword.keys())
        self.list_val_care_idx = []

        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)

        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')

    def set_vis(self):
        if self.cfg_dataset_ver2:
            pass
        else:
            self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
            self.dict_cls_id_to_name = dict()
            for k, v in self.dict_cls_name_to_id.items():
                if v != -1:
                    self.dict_cls_id_to_name[v] = k
            self.dict_cls_name_to_bgr = self.cfg.VIS.CLASS_BGR
            self.dict_cls_name_to_rgb = self.cfg.VIS.CLASS_RGB

    def set_weather_classifier(self):
        self.use_weather_classifier = False
        if not self.cfg.MODEL.get('LoRA', {}).get('MTLoRA', False):
            return

        cfg_weather = self.cfg.MODEL.get('WEATHER_CLASSIFIER', None)
        if cfg_weather is None or not cfg_weather.get('ENABLED', False):
            return

        path_cls = cfg_weather.get('PATH', None)
        if path_cls is None:
            path_cls = self.cfg.GENERAL.get('RESUME', {}).get('IMG_CLS_PATH', None)
        if path_cls is None:
            print('* Warning (Pipeline): WEATHER_CLASSIFIER is enabled but no PATH is provided.')
            return

        try:
            self.weather_classifier = WeatherClsResNet18().cuda().eval()
            load_weights(self.weather_classifier, path_cls, strict=True, lora_align=False)
            for p in self.weather_classifier.parameters():
                p.requires_grad = False
            self.use_weather_classifier = True
        except Exception as e:
            self.weather_classifier = None
            self.use_weather_classifier = False
            print(f'* Warning (Pipeline): failed to load weather classifier ({e}). Fallback to meta weather labels.')

    def _predict_weather_for_batch(self, dict_datum):
        if not self.use_weather_classifier or self.weather_classifier is None:
            return None
        if 'cam_front_img' not in dict_datum:
            return None

        with torch.no_grad():
            cam_front_img = dict_datum['cam_front_img']
            if not isinstance(cam_front_img, torch.Tensor):
                return None
            cam_front_img = cam_front_img.cuda(non_blocking=True)
            dict_pred = self.weather_classifier({'cam_front_img': cam_front_img})
            logits = dict_pred['img_cls_output']
            pred_idx = torch.argmax(logits, dim=1).detach().cpu().tolist()

        pred_weather_list = [self.weather_list[idx] for idx in pred_idx]
        dict_datum['pred_weather_list'] = pred_weather_list
        weather_context = Counter(pred_weather_list).most_common(1)[0][0]
        dict_datum['weather_context'] = weather_context
        return weather_context

    def _resolve_batch_weather(self, dict_datum):
        weather_context = dict_datum.get('weather_context', None)
        if weather_context is not None:
            return weather_context

        pred_weather_list = dict_datum.get('pred_weather_list', None)
        if pred_weather_list is not None and len(pred_weather_list) > 0:
            weather_context = Counter(pred_weather_list).most_common(1)[0][0]
            dict_datum['weather_context'] = weather_context
            return weather_context

        weather_conditions = [sample['desc']['climate'] for sample in dict_datum['meta']]
        assert len(set(weather_conditions)) == 1
        weather_context = weather_conditions[0]
        dict_datum['weather_context'] = weather_context
        return weather_context

    def _count_params(self):

        from models.lora.layers import MTLoRAConv

        total_params = sum(p.numel() for name, p in self.network.named_parameters()
                          if 'lora_A' not in name and 'lora_B' not in name)
        trainable_params = sum(p.numel() for name, p in self.network.named_parameters()
                              if p.requires_grad and 'lora_A' not in name and 'lora_B' not in name)

        if self.open_lora:
            for module in self.network.modules():
                if isinstance(module, MTLoRAConv):
                    lora_count = module.get_actual_lora_params()
                    total_params += lora_count
                    trainable_params += lora_count

        return total_params, trainable_params

    def set_logging(self, path_cfg, is_print_where=True):
        self.is_logging = True
        str_local_time = get_local_time_str()
        str_exp = 'exp_' + str_local_time + '_' + self.cfg.GENERAL.NAME + '_' + self.cfg.GENERAL.get('NAME2','')
        self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, str_exp)
        if is_print_where:
            print(f'* Start logging in {str_exp}')
        if not (os.path.exists(self.path_log)):
            os.makedirs(self.path_log)
        else:
            print('* Exception error (Pipeline): same folder exists, try again')
            exit()

        self.log_train_iter = SummaryWriter(os.path.join(self.path_log, 'train_iter'), comment='iteration')
        self.log_train_epoch = SummaryWriter(os.path.join(self.path_log, 'train_epoch'), comment='epoch')
        self.log_test = SummaryWriter(os.path.join(self.path_log, 'test'), comment='test')
        self.log_iter_start = None

        self.is_save_model = self.cfg.GENERAL.LOGGING.IS_SAVE_MODEL
        try:
            self.interval_epoch_model = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_MODEL
            self.interval_epoch_util = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_UTIL
        except:
            self.interval_epoch_model = 1
            self.interval_epoch_util = 5
            print('* Exception error (Pipeline): check LOGGING.INTERVAL_EPOCH_MODEL/UTIL')
        if self.is_save_model:
            os.makedirs(os.path.join(self.path_log, 'models'))
            os.makedirs(os.path.join(self.path_log, 'utils'))

        name_file_origin = path_cfg.split('/')[-1]
        name_file_cfg = 'config.yml'
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_origin))
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_cfg))

        log_file_path = os.path.join(self.path_log, 'output.log')
        log_file = open(log_file_path, 'w')
        sys.stdout = Tee(log_file)

    def load_state_dict_with_multihead(self, state_dict):
        for key, param in self.network.named_parameters():
            module_name, key_name = key.split('.', 1)
            if '_' not in module_name:
                continue
            prefix, suffix = module_name.rsplit('_', 1)
            if prefix in self.head_names:
                original_key = f'{prefix}.{key_name}'
                if original_key in state_dict.keys():
                    param.data.copy_(state_dict[original_key])
                else:
                    print(f'* Warning: {original_key} not found in state_dict')

    def resume_network(self):
        path_exp = self.cfg.GENERAL.RESUME.get('PATH_EXP', None)
        img_cls_path = self.cfg.GENERAL.RESUME.get('IMG_CLS_PATH', None)
        if img_cls_path is not None:
            load_weights(self.network.img_cls, img_cls_path)
        if path_exp is None:
            print('* Exception error (Pipeline): check RESUME.PATH_EXP')
            return
        path_state_dict = os.path.join(path_exp, 'utils')
        epoch = self.cfg.GENERAL.RESUME.START_EP
        list_epochs = sorted(list(map(lambda x: int(x.split('.')[0].split('_')[1]), os.listdir(path_state_dict))))
        epoch = list_epochs[-1] if epoch is None else epoch

        path_state_dict = os.path.join(path_state_dict, f'util_{epoch}.pt')

        try:
            print('* Start resume, path_state_dict =  ', path_state_dict)
            state_dict = torch.load(path_state_dict)
            self.epoch_start = epoch + 1
            if self.open_multi_head and self.cfg.GENERAL.RESUME.get('COPY_MULTI_HEAD', False):
                self.network.load_state_dict(state_dict['model_state_dict'], strict=False)
                self.load_state_dict_with_multihead(state_dict['model_state_dict'])
                self.log_iter_start = state_dict['idx_log_iter']
            else:
                self.network.load_state_dict(state_dict['model_state_dict'])
                self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                self.log_iter_start = state_dict['idx_log_iter']
        except:
            path_state_dict = os.path.join(path_exp, f'models/model_{epoch}.pt')
            print('* Start Load Model, path_state_dict =  ', path_state_dict)
            state_dict = torch.load(path_state_dict)
            self.network.load_state_dict(state_dict)

        if self.cfg.GENERAL.RESUME.get('IS_FREEZE_BACKBONE', False):
            for name, module in self.network.named_children():
                if self.open_multi_head and any(weather in name for weather in self.weather_list):
                    name = name.rsplit('_', 1)[0]
                if not name in self.head_names:
                    for param_name, param in module.named_parameters():

                        param.requires_grad = False

                    if self.cfg.GENERAL.RESUME.get('IS_FREEZE_BATCHNORM', False):
                        for sub_module in module.modules():
                            if isinstance(sub_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                                sub_module.momentum = 0
                    else:
                        for sub_module in module.modules():
                            if isinstance(sub_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                                for param_name, param in sub_module.named_parameters():

                                    param.requires_grad = True

        if ('scheduler_state_dict' in state_dict.keys()) and (not (self.scheduler is None)) and\
            not self.cfg.GENERAL.RESUME.get('COPY_MULTI_HEAD', False):
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        else:
            print('* Schduler is started from vanilla')

        list_copy_dirs = ['train_epoch', 'train_iter', 'test', 'test_kitti']
        if (self.cfg.GENERAL.RESUME.IS_COPY_LOGS) and (self.is_logging):
            for copy_dir in list_copy_dirs:
                shutil.copytree(os.path.join(path_exp, copy_dir),\
                    os.path.join(self.path_log, copy_dir), dirs_exist_ok=True)

        return

    def train_network(self, is_shuffle=True):
        self.network.train()
        is_shuffle = is_shuffle and (self.cfg.DATASET.get('is_shuffle', True))
        if self.cfg.DATASET.get('sort_by_weather', False):
            assert self.dataset_train.batch_sampler is not None
            data_loader_train = torch.utils.data.DataLoader(self.dataset_train,\
                collate_fn = self.dataset_train.collate_fn,
                batch_sampler = self.dataset_train.batch_sampler,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        else:
            data_loader_train = torch.utils.data.DataLoader(self.dataset_train,\
                batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = is_shuffle,\
                collate_fn = self.dataset_train.collate_fn,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, drop_last = True)

        epoch_start = self.epoch_start
        epoch_end = self.cfg.OPTIMIZER.MAX_EPOCH

        if self.is_logging:
            idx_log_iter = 0 if self.log_iter_start is None else self.log_iter_start

        if self.optim_fastai:
            accumulated_iter = 0
            cfg_optim = self.cfg.OPTIMIZER
            use_amp = cfg_optim.get('USE_AMP', False)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=cfg_optim.get('LOSS_SCALE_FP16', 2.0**16))

        for epoch in range(epoch_start, epoch_end):
            print(f'* Training epoch = {epoch}/{epoch_end-1}')
            if self.is_logging:
                print(f'* Logging path = {self.path_log}')

            self.network.train()
            self.network.training = True
            avg_loss = []
            avg_multi_loss = {}
            pbar = tqdm(data_loader_train, desc='Training')
            for idx_iter, dict_datum in enumerate(pbar):
                self._predict_weather_for_batch(dict_datum)
                weather = self._resolve_batch_weather(dict_datum)

                if self.mtoptim:
                    optimizer = self.optimizer[weather]
                    scheduler = self.scheduler[weather]
                else:
                    optimizer = self.optimizer
                    scheduler = self.scheduler

                if self.optim_fastai:
                    scheduler.step(accumulated_iter, epoch)

                if self.distil:
                    with torch.no_grad():
                        dict_datum = self.distil_model(dict_datum)
                        dict_datum['ldr_bev_feat'] = dict_datum['spatial_features_2d']

                dict_net = self.network(dict_datum)

                if self.get_loss_from == 'head':
                    if self.open_multi_head:
                        weather = self._resolve_batch_weather(dict_net)
                        head_module = getattr(self.network, f'head_{weather}', None)
                        loss = head_module.loss(dict_net)
                        if weather not in avg_multi_loss.keys():
                            avg_multi_loss[weather] = []
                        avg_multi_loss[weather].append(loss.cpu().detach().item())
                    else:
                        loss = self.network.head.loss(dict_net)
                        if 'any' not in avg_multi_loss.keys():
                            avg_multi_loss['any'] = []
                            avg_multi_loss['any'].append(loss.cpu().detach().item())
                elif self.get_loss_from == 'detector':
                    loss = self.network.loss(dict_net)
                    if self.open_multi_head:
                        weather = self._resolve_batch_weather(dict_net)
                        if weather not in avg_multi_loss.keys():
                            avg_multi_loss[weather] = []
                        avg_multi_loss[weather].append(loss.cpu().detach().item())
                    else:
                        if 'any' not in avg_multi_loss.keys():
                            avg_multi_loss['any'] = []
                        avg_multi_loss['any'].append(loss.cpu().detach().item())

                try:
                    log_avg_loss = loss.cpu().detach().item()
                except:
                    log_avg_loss = loss
                avg_loss.append(log_avg_loss)

                if self.optim_fastai:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(self.network.parameters(), cfg_optim.GRAD_NORM_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    accumulated_iter += 1
                else:
                    if loss == 0.:
                        pass

                    elif torch.isfinite(loss):
                        loss.backward()
                    else:
                        print('* Exception error (pipeline): nan or inf loss happend')
                        print('* Meta: ', dict_datum['meta'])
                    if self.open_lora and self.ada_lora:
                        for name, module in self.network.named_modules():
                            if isinstance(module, MTLoRABase) and hasattr(module, 'calculate_grad'):
                                module.calculate_grad(weather)

                    optimizer.step()
                    if not (scheduler is None):
                        scheduler.step()

                optimizer.zero_grad()

                if self.is_logging:
                    dict_logging = dict_net['logging']
                    idx_log_iter +=1
                    for k, v in dict_logging.items():
                        self.log_train_iter.add_scalar(f'train/{k}', v, idx_log_iter)
                    if not (scheduler is None):
                        if self.optim_fastai:
                            lr = float(optimizer.lr)
                            self.log_train_iter.add_scalar(f'train/learning_rate', lr, idx_log_iter)
                        else:
                            lr = scheduler.get_last_lr()[0]
                            self.log_train_iter.add_scalar(f'train/learning_rate', lr, idx_log_iter)

                if 'pointer' in dict_datum.keys():
                    for dict_item in dict_datum['pointer']:
                        for k in dict_item.keys():
                            if k != 'meta':
                                dict_item[k] = None
                for temp_key in dict_datum.keys():
                    dict_datum[temp_key] = None

                if not (scheduler is None):
                    if self.optim_fastai:
                        lr = float(optimizer.lr)
                        self.log_train_iter.add_scalar(f'train/learning_rate', lr, idx_log_iter)
                    else:
                        lr = scheduler.get_last_lr()[0]
                        self.log_train_iter.add_scalar(f'train/learning_rate', lr, idx_log_iter)
                else:
                    lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "lr": f"{lr:.3e}",
                    "loss": f"{avg_loss[-1]:.3e}",

                })

            if self.open_lora and self.ada_lora and epoch <= self.ada_end_epoch and (epoch + 1) % self.ada_epoch_freq == 0:
                for name, module in self.network.named_modules():
                    if isinstance(module, MTLoRABase) and hasattr(module, 'adaptive_lora_rank'):
                        module.adaptive_lora_rank(k = self.cfg.MODEL.LoRA.get('AdaLoRA_prune', 2))
                        print(f'{name} adaptive lora rank: {module.current_rank()}')

            if self.is_save_model:

                path_dict_model = os.path.join(self.path_log, 'models', f'model_{epoch}.pt')
                path_dict_util = os.path.join(self.path_log, 'utils', f'util_{epoch}.pt')

                if (epoch+1) % self.interval_epoch_model == 0:
                    torch.save(self.network.state_dict(), path_dict_model)
                if (epoch+1) % self.interval_epoch_util == 0:
                    dict_util = {
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': {weather: optim.state_dict() for weather, optim in self.optimizer.items()} if self.mtoptim else self.optimizer.state_dict(),
                        'idx_log_iter': idx_log_iter,
                    }
                    if self.optim_fastai:
                        dict_util.update({'it': accumulated_iter})
                    else:
                        if not (self.scheduler is None):
                            dict_util.update({'scheduler_state_dict': {weather: sched.state_dict() for weather, sched in self.scheduler.items()} if self.mtoptim else self.scheduler.state_dict()})
                    torch.save(dict_util, path_dict_util)

            if self.is_logging:
                self.log_train_epoch.add_scalar(f'train/avg_loss', np.mean(avg_loss), epoch)

            if self.is_validate:
                if ((epoch + 1) % self.val_per_epoch_full) == 0:
                    self.validate_kitti_conditional(epoch=epoch, list_conf_thr=[0.3], is_subset=False, is_print_memory=False, save_best_model=True)
                else:
                    if self.is_consider_subset:
                        if ((epoch + 1) % self.val_per_epoch_subset) == 0:
                            self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr, is_subset=True)

    def load_dict_model(self, path_dict_model, is_strict=False):
        pt_dict_model = torch.load(path_dict_model)
        self.network.load_state_dict(pt_dict_model, strict=is_strict)

    def vis_infer(self, sample_indices, conf_thr=0.7, is_nms=True, vis_mode=['lpc', 'spcube', 'cube'], is_train=False):

        self.network.eval()

        if is_train:
            dataset_loaded = self.dataset_train
        else:
            dataset_loaded = self.dataset_test
        subset = Subset(dataset_loaded, sample_indices)
        data_loader = torch.utils.data.DataLoader(subset,
                batch_size = 1, shuffle = False,
                collate_fn = self.dataset_test.collate_fn,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        for dict_datum in data_loader:
            self._predict_weather_for_batch(dict_datum)
            dict_out = self.network(dict_datum)
            dict_out = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms)

            pc_lidar = dict_datum['ldr64']

            labels = dict_out['label'][0]
            list_obj_label = []
            for label_obj in labels:
                cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
                obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                list_obj_label.append(obj)

            list_obj_pred = []
            list_cls_pred = []
            if dict_datum['pp_num_bbox'] == 0:
                pass
            else:
                pp_cls = dict_datum['pp_cls']
                for idx_pred, pred_obj in enumerate(dict_datum['pp_bbox']):
                    conf_score, xc, yc, zc, xl, yl, zl, rot = pred_obj
                    obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                    list_obj_pred.append(obj)
                    list_cls_pred.append('Sedan')

            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [6, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_label = [[0, 0, 0] for _ in range(len(lines))]
            list_line_set_label = []
            list_line_set_pred = []
            for label_obj in list_obj_label:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_label)
                list_line_set_label.append(line_set)

            for idx_pred, pred_obj in enumerate(list_obj_pred):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)

                colors_pred = [[1.,0.,0.] for _ in range(len(lines))]
                line_set.colors = o3d.utility.Vector3dVector(colors_pred)
                list_line_set_pred.append(line_set)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred)

        return list_obj_label, list_obj_pred

    def validate_kitti(self, epoch=None, list_conf_thr=None, is_subset=False):
        self.network.training=False
        self.network.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name

        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='* Test (Subset): ')
            log_header = 'val_sub'
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(self.dataset_test), desc='* Test (Total): ')
            log_header = 'val_tot'

        data_loader = torch.utils.data.DataLoader(self.dataset_test,\
                batch_size=1, shuffle=is_shuffle, collate_fn=self.dataset_test.collate_fn,\
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)

        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
                f.write('')
            f.close()

        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break

            try:
                self._predict_weather_for_batch(dict_datum)
                dict_out = self.network(dict_datum)
                is_feature_inferenced = True
            except:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False

            idx_name = str(idx_datum).zfill(6)

            for conf_thr in list_conf_thr:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'val.txt'
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]

                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out)
                if dict_out is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out = dict_datum_to_kitti(self, dict_out)

                if len(dict_out['kitti_gt']) == 0:
                    pass
                else:

                    for idx_label, label in enumerate(dict_out['kitti_gt']):
                        open_mode = 'w' if idx_label == 0 else 'a'
                        with open(labels_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(label+'\n')

                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out['kitti_desc'])

                    for idx_pred, pred in enumerate(dict_out['kitti_pred']):
                        open_mode = 'w' if idx_pred == 0 else 'a'
                        with open(preds_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(pred+'\n')

                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)

            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        for conf_thr in list_conf_thr:
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
            split_path = path_dir + f'/{conf_thr}/' + 'val.txt'

            dt_annos = kitti.get_label_annos(preds_dir)
            val_ids = read_imageset_file(split_path)
            gt_annos = kitti.get_label_annos(labels_dir, val_ids)

            list_metrics = []
            for idx_cls_val in self.list_val_care_idx:
                if self.is_validation_updated:

                    dict_metrics, result_log = get_official_eval_result_revised(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                else:
                    dict_metrics, result_log = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                print(f'-----conf{conf_thr}-----')
                print(result_log)
                list_metrics.append(dict_metrics)

            for dict_metrics in list_metrics:
                cls_name = dict_metrics['cls']
                ious = dict_metrics['iou']
                bevs = dict_metrics['bev']
                ap3ds = dict_metrics['3d']
                self.log_test.add_scalars(f'{log_header}/BEV_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': bevs[0],
                    f'iou_{ious[1]}_{cls_name}': bevs[1],
                    f'iou_{ious[2]}_{cls_name}': bevs[2],
                }, epoch)
                self.log_test.add_scalars(f'{log_header}/3D_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': ap3ds[0],
                    f'iou_{ious[1]}_{cls_name}': ap3ds[1],
                    f'iou_{ious[2]}_{cls_name}': ap3ds[2],
                }, epoch)

    def validate_kitti_conditional(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, save_best_model=False, keep_intermediate=False):
        self.network.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name

        road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
        time_cond_list = ['day', 'night']
        weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='Test (Subset): ')
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')

        data_loader = torch.utils.data.DataLoader(self.dataset_test,\
                batch_size = 1, shuffle = is_shuffle, collate_fn = self.dataset_test.collate_fn,\
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)

            os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                f.write('')

            for road_cond in road_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                    f.write('')

            for time_cond in time_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                    f.write('')

            for weather_cond in weather_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                    f.write('')

            pred_dir_list = []
            label_dir_list = []
            desc_dir_list = []
            split_path_list = []

            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
            list_dir = [preds_dir, labels_dir, desc_dir]
            split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

            for temp_dir in list_dir:
                os.makedirs(temp_dir, exist_ok=True)

            pred_dir_list.append(preds_dir)
            label_dir_list.append(labels_dir)
            desc_dir_list.append(desc_dir)
            split_path_list.append(split_path)

            for road_cond in road_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

            for time_cond in time_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

            for weather_cond in weather_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break

            try:
                self._predict_weather_for_batch(dict_datum)
                dict_out = self.network(dict_datum)
                is_feature_inferenced = True
            except:
                dict_out = dict_datum
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False
                if dict_out['meta'] is None:
                    continue

            if is_print_memory:
                print('max_memory: ', torch.cuda.max_memory_allocated(device='cuda'))

            idx_name = str(idx_datum).zfill(6)

            road_cond_tag, time_cond_tag, weather_cond_tag =\
                dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']

            for conf_thr in list_conf_thr:

                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                split_path_weather =path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'

                os.makedirs(labels_dir_road, exist_ok=True)
                os.makedirs(labels_dir_time, exist_ok=True)
                os.makedirs(labels_dir_weather, exist_ok=True)
                os.makedirs(desc_dir_road, exist_ok=True)
                os.makedirs(desc_dir_time, exist_ok=True)
                os.makedirs(desc_dir_weather, exist_ok=True)
                os.makedirs(preds_dir_road, exist_ok=True)
                os.makedirs(preds_dir_time, exist_ok=True)
                os.makedirs(preds_dir_weather, exist_ok=True)

                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]

                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out)

                if dict_out_current is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                if len(dict_out_current['kitti_gt']) == 0:
                    pass
                else:

                    for idx_label, label in enumerate(dict_out_current['kitti_gt']):
                        if idx_label == 0:
                            mode = 'w'
                        else:
                            mode = 'a'

                        with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')

                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])

                    if len(dict_out_current['kitti_pred']) == 0:
                        with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                    else:
                        for idx_pred, pred in enumerate(dict_out_current['kitti_pred']):
                            if idx_pred == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')

                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)
                    with open(split_path_road, 'a') as f:
                        f.write(str_log)
                    with open(split_path_time, 'a') as f:
                        f.write(str_log)
                    with open(split_path_weather, 'a') as f:
                        f.write(str_log)

            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        for conf_thr in list_conf_thr:
            for cond_list in [road_cond_list, time_cond_list, weather_cond_list]:
                for idx_cls_val in self.list_val_care_idx:
                    cond_metrics_dict = {}
                    for condition in ['all'] + cond_list:
                        try:
                            preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                            labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                            desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                            split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                            dt_annos = kitti.get_label_annos(preds_dir)
                            val_ids = read_imageset_file(split_path)
                            gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                            if self.is_validation_updated:

                                dict_metrics, result = get_official_eval_result_revised(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                            else:
                                dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                            cond_metrics_dict[condition] = dict_metrics
                        except:
                            print(f'* Exception error (Pipeline): Samples for the codition{condition} are not found')
                            cond_metrics_dict[condition] = None
                    for iou_id, iou in enumerate(cond_metrics_dict['all']['iou']):
                        print(f'Conf thr: {conf_thr} | Cls: {cond_metrics_dict["all"]["cls"]} | IoU: {iou}')
                        print('=' * 70)

                        cond_names = ['all', 'avg'] + cond_list

                        bev_con, td_con = [], []
                        for condition in cond_names:
                            if condition == 'avg':
                                avg_bev = sum(cond_metrics_dict[c]['bev'][iou_id] if cond_metrics_dict[c] else 0 for c in cond_list)/len(cond_list) if cond_list else 0
                                avg_3d = sum(cond_metrics_dict[c]['3d'][iou_id] if cond_metrics_dict[c] else 0 for c in cond_list)/len(cond_list) if cond_list else 0
                                bev_con.append(avg_bev)
                                td_con.append(avg_3d)
                            else:
                                bev_con.append(cond_metrics_dict[condition]['bev'][iou_id] if cond_metrics_dict[condition] else 0)
                                td_con.append(cond_metrics_dict[condition]['3d'][iou_id] if cond_metrics_dict[condition] else 0)

                        header = f"{'Condition':<12}" + "".join([f"{name:<12}" for name in cond_names])
                        print(f"{' ':<12}" + "".join([f"{name:<12}" for name in cond_names]))
                        bev_row = f"{'BEV:':<12}" + "".join([f"{val:<12.4f}" for val in bev_con])
                        print(bev_row)
                        td_row = f"{'3D:':<12}" + "".join([f"{val:<12.4f}" for val in td_con])
                        print(td_row)

                        print('-' * 70)
                        print("\n")

                        if save_best_model and cond_list == weather_cond_list and cond_metrics_dict["all"]["cls"] == 'sed':
                            for weather in cond_list:
                                try:
                                    if iou not in self.best_weather_results:
                                        self.best_weather_results[iou] = {}
                                        self.best_weather_models[iou] = {}
                                    current_metric = cond_metrics_dict[weather]['3d'][iou_id]
                                    if (weather not in self.best_weather_results[iou] or
                                        self.best_weather_results[iou][weather] < current_metric):

                                        self.best_weather_results[iou][weather] = current_metric

                                        weather_model_state = {
                                            k: v.clone().detach().cpu()
                                            for k, v in self.network.state_dict().items()
                                            if weather in k
                                        }
                                        self.best_weather_models[iou][weather] = weather_model_state
                                except:
                                    print(f'* Exception error (Pipeline): Samples for the weather{weather} are not found')
                                    continue

        if not keep_intermediate:
            shutil.rmtree(path_dir)
            print(f"* Cleaned intermediate files in {path_dir}")
        path_check = os.path.join(path_dir, 'Conf_thr', 'complete_results.txt')
        print(f'* Check {path_check}')

    def load_best_model(self, iou=0.3):
        if not hasattr(self, 'best_weather_models'):
            return
        print('* load best model for multi weather')
        try:
            print(f'IOU best {iou}: {self.best_weather_results[iou]}')
        except:
            iou=f'{iou}'
            print(f'IOU best {iou}: {self.best_weather_results[iou]}')

        for weather, weather_dict in self.best_weather_models[iou].items():

            filtered_dict = {
                k: v for k, v in weather_dict.items()

            }

            self.network.load_state_dict(filtered_dict, strict=False)

    def validata_best_model(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, best_iou=0.3):
        original_state_dict = copy.deepcopy(self.network.state_dict())
        self.load_best_model(best_iou)
        torch.save(self.network.state_dict(), os.path.join(self.path_log, 'models', f'model_best_{epoch}_iou{best_iou}.pt'))
        self.validate_kitti_conditional(epoch=epoch, list_conf_thr=list_conf_thr, is_subset=is_subset, is_print_memory=is_print_memory, save_best_model=False)
        self.network.load_state_dict(original_state_dict, strict=True)
