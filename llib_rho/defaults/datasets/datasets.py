import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass

############################# DATASETS ############################
    
@dataclass
class Behave:
    data_folder: str = 'datasets/original/behave'
    image_folder: str = 'images'
    bev_folder: str = 'bev'
    openpose_folder: str = 'keypoints/openpose'
    vitpose_folder: str = 'keypoints/vitpose'
    vitposeplus_folder: str = 'keypoints/vitposeplus'
    pose_folder='pose_pred'
    resnet_folder ='resnet_feat'
    pseudogt_folder: str = 'gt'
    image_format: str = 'jpg'
    overfit: bool = False
    overfit_num_samples: int = 12    

## Reference
# @dataclass
# class HI4D:
#     original_data_folder: str  = 'datasets/original/Hi4D'
#     processed_data_folder: str = 'datasets/processed/Hi4D'
#     image_folder: str = 'images'
#     bev_folder: str = 'bev'
#     openpose_folder: str = 'keypoints/keypoints'
#     vitpose_folder: str = 'keypoints/vitposeplus'
#     image_format: str = 'jpg'
#     overfit: bool = False
#     overfit_num_samples: int = 12
#     load_single_camera: bool = False
#     load_from_scratch_single_camera: bool = False
#     load_unit_glob_and_transl: bool = False
#     features: DatasetFeatures = DatasetFeatures(
#         is_itw = False, 
#         has_dhhc_sig = True,
#         has_op_kpts = True,
#     )

# @dataclass
# class Demo:
#     original_data_folder: str  = ''
#     number_of_regions: int = 75
#     image_folder: str = 'images'
#     bev_folder: str = 'bev'
#     openpose_folder: str = 'keypoints/keypoints'
#     vitpose_folder: str = 'vitpose'
#     image_format: str = 'png'
#     image_name_select: str = ''
#     has_gt_contact_annotation: bool = False
#     imar_vision_datasets_tools_folder: str = 'essentials/imar_vision_datasets_tools' #'datasets/original'
#     unique_keypoint_match: bool = True
