import os
from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

import argparse

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--config', type=str, default='./configs/cfg_RTNH_wide_lambda.yml', help='Path to the configuration file')
parser.add_argument('--gpus', type=str, default='0', help='Specify which CUDA devices are visible (e.g., "0,1,2")')
parser.add_argument('--seed', type=int, default=None, help='Specify the random seed')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

PATH_CONFIG = args.config

if __name__ == '__main__':
    pline = PipelineDetection_v1_0(path_cfg=PATH_CONFIG, mode='train')

    import shutil
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))

    if pline.cfg.get('before_val', True):
        pline.validate_kitti_conditional(list_conf_thr=[0.3], is_subset=False, is_print_memory=False, save_best_model=True)
    pline.train_network()

    if pline.open_lora:
        pline.validata_best_model(list_conf_thr=[0.3], is_subset=False, is_print_memory=False, best_iou=0.5)
    else:
        pline.validate_kitti_conditional(list_conf_thr=[0.3], is_subset=False, is_print_memory=False, save_best_model=False)
