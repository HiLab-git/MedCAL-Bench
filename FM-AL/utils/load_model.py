import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
from Foundation_models import sam_model_registry




def load_sam_image_encoder(vit_type):
    sam = sam_model_registry[vit_type](checkpoint=sam_checkpoint_path[vit_type])
    return sam.image_encoder, sam.preprocess

def load_medsam_image_encoder(vit_type):
    sam = sam_model_registry[vit_type](checkpoint=sam_checkpoint_path[vit_type])

def load_sam2_image_encoder(config_file, weight_file):
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = OmegaConf.create(cfg)

    image_encoder_cfg = cfg.model.image_encoder

    image_encoder = instantiate(image_encoder_cfg)

    if weight_file:
        state_dict = torch.load(weight_file)["model"]
        model_state_dict = image_encoder.state_dict()

        updated_state_dict = {}
        for param_name, param_tensor in model_state_dict.items():
            if f"image_encoder.{param_name}" in state_dict:
                updated_state_dict[param_name] = state_dict[
                    f"image_encoder.{param_name}"
                ]
            else:
                print(f"Warning: Parameter not found in state_dict: {param_name}")

        missing_keys, unexpected_keys = image_encoder.load_state_dict(
            updated_state_dict, strict=False
        )
        if missing_keys:
            print(f"Warning: Missing keys when loading weights: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading weights: {unexpected_keys}")

    return image_encoder


sam_checkpoint_path = {
    "vit_b": "./Foundation_models/checkpoints/sam_vit_b_01ec64.pth",
    "vit_h": "./Foundation_models/checkpoints/sam_vit_h_4b8939.pth",
    "medsam_vit_b": "./Foundation_models/checkpoints/medsam_vit_b.pth"
}

sam2_config = {
    "2b+": {
        "config_file": "./Foundation_models/model_cfg/sam2_hiera_b+.yaml",
        "weight_file": "./Foundation_models/checkpoints/sam2_hiera_base_plus.pt",
    },
    "2l": {
        "config_file": "./Foundation_models/model_cfg/sam2_hiera_l.yaml",
        "weight_file": "./Foundation_models/checkpoints/sam2_hiera_large.pt",
    },
    "2s": {
        "config_file": "./Foundation_models/model_cfg/sam2_hiera_s.yaml",
        "weight_file": "./Foundation_models/checkpoints/sam2_hiera_small.pt",
    },
    "2t": {
        "config_file": "./Foundation_models/model_cfg/sam2_hiera_t.yaml",
        "weight_file": "./Foundation_models/checkpoints/sam2_hiera_tiny.pt",
    },
    "2.1b+": {
        "config_file": "./Foundation_models/model_cfg/sam2.1_hiera_b+.yaml",
        "weight_file": "./Foundation_models/checkpoints/sam2.1_hiera_base_plus.pt",
    },
    "2.1l": {
        "config_file": "./Foundation_models/model_cfg/sam2.1_hiera_l.yaml",
        "weight_file": "./Foundation_models/checkpoints/sam2.1_hiera_large.pt",
    },
    "2.1s": {
        "config_file": "./Foundation_models/model_cfg/sam2.1_hiera_s.yaml",
        "weight_file": "./Foundation_models/checkpoints/sam2.1_hiera_small.pt",
    },
    "2.1t": {
        "config_file": "./Foundation_models/model_cfg/sam2.1_hiera_t.yaml",
        "weight_file": "./Foundation_models/checkpoints/sam2.1_hiera_tiny.pt",
    },
}
