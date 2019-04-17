import os.path as osp
import sys

module_dir = osp.dirname(__file__)
src_dir = osp.dirname(module_dir)
sys.path.append(osp.join(src_dir, "external", "coco", "PythonAPI"))
