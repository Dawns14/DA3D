from .rdr_base import RadarBase
from .ldr_base import LidarBase
from .pvrcnn_pp import PVRCNNPlusPlus
from .second_net import SECONDNet
from .pp_rlf import PointPillar_RLF
from .rl_3df_gate import RL3DF_gate

def build_skeleton(cfg):
    return __all__[cfg.MODEL.SKELETON](cfg)

__all__ = {
    'RadarBase': RadarBase,
    'LidarBase': LidarBase,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'SECONDNet': SECONDNet,
    'PointPillar_RLF':PointPillar_RLF,
    'RL3DF_gate': RL3DF_gate,
}
