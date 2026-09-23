import os
import sys

# Ensure self-contained NeMO submodule is discovered before importing bundle_nemo modules
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_pkg_dir)
_submodule_nemo_src = os.path.join(_repo_root, "NeMO", "src")
if os.path.isdir(_submodule_nemo_src) and _submodule_nemo_src not in sys.path:
    sys.path.insert(0, _submodule_nemo_src)

from .tracker import BundleNeMOTracker, crop_masked_object
from .memory_bank import AlignedDynamicNeMOMemoryBank
from .correspondence import NeMOCorrespondenceEngine
from .optimizer import BundleNeMOOptimizer
from .fusion import CanonicalObjectFusion
from .scale_estimator import AutomaticScaleEstimator
from .zero_gt_alignment import ZeroGTAlignmentEngine
from .pose_graph import KeyframePoseGraph

__all__ = [
    'BundleNeMOTracker',
    'crop_masked_object',
    'AlignedDynamicNeMOMemoryBank',
    'NeMOCorrespondenceEngine',
    'BundleNeMOOptimizer',
    'CanonicalObjectFusion',
    'AutomaticScaleEstimator',
    'ZeroGTAlignmentEngine',
    'KeyframePoseGraph'
]
