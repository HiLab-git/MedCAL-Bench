import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    isfile,
    save_json,
    maybe_mkdir_p,
)
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import (
    ContrastTransform,
    BGContrast,
)
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import (
    ApplyRandomBinaryOperatorTransform,
)
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import (
    MoveSegAsOneHotToDataTransform,
)
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import (
    Convert3DTo2DTransform,
    Convert2DTo3DTransform,
)
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import (
    ConvertSegmentationToRegionsTransform,
)
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import json


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2.0 * np.pi, rot_x)
    rot_y = min(90 / 360 * 2.0 * np.pi, rot_y)
    rot_z = min(90 / 360 * 2.0 * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d

    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0
        )
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0
        )
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0
        )
    elif len(coords) == 2:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0
        )
    final_shape /= min(scale_range)
    return final_shape.astype(int)


class ConfigurationManager(object):
    def __init__(self, configuration_dict: dict):
        self.configuration = configuration_dict

    @property
    def data_identifier(self) -> str:
        return self.configuration["data_identifier"]

    @property
    def preprocessor_name(self) -> str:
        return self.configuration["preprocessor_name"]

    @property
    def batch_size(self) -> int:
        return self.configuration["batch_size"]

    @property
    def patch_size(self) -> List[int]:
        return self.configuration["patch_size"]

    @property
    def median_image_size_in_voxels(self) -> List[int]:
        return self.configuration["median_image_size_in_voxels"]

    @property
    def spacing(self) -> List[float]:
        return self.configuration["spacing"]

    @property
    def normalization_schemes(self) -> List[str]:
        return self.configuration["normalization_schemes"]

    @property
    def use_mask_for_norm(self) -> List[bool]:
        return self.configuration["use_mask_for_norm"]

    @property
    def network_arch_class_name(self) -> str:
        return self.configuration["architecture"]["network_class_name"]

    @property
    def network_arch_init_kwargs(self) -> dict:
        return self.configuration["architecture"]["arch_kwargs"]

    @property
    def network_arch_init_kwargs_req_import(self) -> Union[Tuple[str, ...], List[str]]:
        return self.configuration["architecture"]["_kw_requires_import"]

    @property
    def pool_op_kernel_sizes(self) -> Tuple[Tuple[int, ...], ...]:
        return self.configuration["architecture"]["arch_kwargs"]["strides"]

    @property
    def batch_dice(self) -> bool:
        return self.configuration["batch_dice"]

    @property
    def next_stage_names(self) -> Union[List[str], None]:
        ret = self.configuration.get("next_stage")
        if ret is not None:
            if isinstance(ret, str):
                ret = [ret]
        return ret

    @property
    def previous_stage_name(self) -> Union[str, None]:
        return self.configuration.get("previous_stage")


class Transform_generator(object):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "rb+") as f:
            self.configuration_manager = ConfigurationManager(json.load(f))
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.enable_deep_supervision = True

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi)
            else:
                rotation_for_DA = (
                    -180.0 / 360 * 2.0 * np.pi,
                    180.0 / 360 * 2.0 * np.pi,
                )
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > 3
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (
                    -180.0 / 360 * 2.0 * np.pi,
                    180.0 / 360 * 2.0 * np.pi,
                )
            else:
                rotation_for_DA = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(
            patch_size[-dim:],
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.85, 1.25),
        )
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        # self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        # self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True
                ),
                apply_probability=0.1,
            )
        )
        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True,
                ),
                apply_probability=0.2,
            )
        )
        transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.15,
            )
        )
        transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.15,
            )
        )
        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                apply_probability=0.25,
            )
        )
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.1,
            )
        )
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.3,
            )
        )
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[
                        i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]
                    ],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        transforms.append(RemoveLabelTansform(-1, 0))
        if is_cascaded:
            assert (
                foreground_labels is not None
            ), "We need foreground_labels for cascade augmentations"
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True,
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1,
                    ),
                    apply_probability=0.4,
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1,
                    ),
                    apply_probability=0.2,
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=(
                        list(regions) + [ignore_label]
                        if ignore_label is not None
                        else regions
                    ),
                    channel_in_seg=0,
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(
                DownsampleSegForDSTransform(ds_scales=deep_supervision_scales)
            )

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True,
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=(
                        list(regions) + [ignore_label]
                        if ignore_label is not None
                        else regions
                    ),
                    channel_in_seg=0,
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(
                DownsampleSegForDSTransform(ds_scales=deep_supervision_scales)
            )
        return ComposeTransforms(transforms)

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales



    def tr_val_trans(self):
        patch_size = self.configuration_manager.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        

        deep_supervision_scales = self._get_deep_supervision_scales()

        tr_transforms = self.get_training_transforms(
            patch_size=patch_size,
            rotation_for_DA=rotation_for_DA,
            deep_supervision_scales=deep_supervision_scales,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=1,
            regions=None,
            ignore_label=None
        )

        # validation pipeline
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales=deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=1,
            regions=None,
            ignore_label=None
        )
        

        return tr_transforms, val_transforms
    


if __name__ == "__main__":
    trans = Transform_generator(
        config_path="/media/ubuntu/FM-AL/data/dataset/config/Spleen.json"
    )
    train_trans, val_trans = trans.tr_val_trans()
