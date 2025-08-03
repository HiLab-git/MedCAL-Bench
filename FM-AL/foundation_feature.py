import argparse
import numpy as np
import pandas as pd
import os
import pickle
import SimpleITK as sitk
from PIL import Image
import torch
from tqdm import tqdm
from typing import *
from numpy import ndarray
import clip
from PIL import Image
from utils.load_model import (
    load_sam_image_encoder,
    load_sam2_image_encoder,
)
from utils.load_model import sam2_config, sam_checkpoint_path

# from transformers import AutoModel, AutoImageProcessor
from transformers import AutoModel, AutoProcessor
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from PIL import Image
from Foundation_models import (
    ResizeLongestSide,
    SAM2Transforms,
    MedCLIPVisionModel,
    MedCLIPVisionModelViT,
    MedCLIPProcessor,
)


class Feature_extractor:
    def __init__(
        self,
        base_dir="../data/dataset/Heart",
    ):
        self.base_dir = base_dir
        csv_pth = os.path.join(base_dir, "splits/train.csv")
        df = pd.read_csv(csv_pth)
        self.image_pth = df["image_pth"]

    def read_image(self, image_pth):
        _, suffix = os.path.splitext(image_pth)
        if suffix == ".pkl":
            with open(image_pth, "rb") as f:
                image = pickle.load(f)
        elif suffix == ".gz":
            image = sitk.GetArrayFromImage(sitk.ReadImage(image))
        elif suffix == ".npy":
            image = np.load(image_pth)
        else:
            raise ValueError(f"Unsupported image format: {image_pth}")
        return image

    def extract_sam_feature(self, model_type="vit_h") -> Tuple[ndarray, List[str]]:
        image_encoder, preprocess = load_sam_image_encoder(vit_type=model_type)
        image_encoder = image_encoder.cuda()
        resize_transform = ResizeLongestSide(image_encoder.img_size)
        for image_pth in tqdm(self.image_pth):
            feats = []
            name_list = []
            image = self.read_image(image_pth).squeeze()
            image = resize_transform.apply_image(image)
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            image = preprocess(image).cuda()
            with torch.no_grad():
                image_feature = image_encoder(image)
                image_feature = image_feature.mean(dim=[2, 3])

            feats.append(image_feature.squeeze().detach().cpu().numpy())

            name_list.append(image_pth)
            self.save_feature(feats, name_list, model="sam", model_type=model_type)

        return feats, name_list

    def extract_sam2_feature(self, model_type="2b+") -> Tuple[ndarray, List[str]]:
        feats = []
        name_list = self.image_pth
        transform = SAM2Transforms(
            resolution=512,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        config = sam2_config[model_type]
        image_encoder = load_sam2_image_encoder(**config).cuda()
        for image_pth in tqdm(self.image_pth):
            feats = []
            name_list = []
            image = self.read_image(image_pth).squeeze()
            if image.ndim == 2 or image.shape[-1] == 1:
                image = (
                    np.expand_dims(image, axis=-1).repeat(3, axis=-1).astype(np.float32)
                )
            image = transform(image).unsqueeze(0).cuda()
            with torch.no_grad():
                image_feature = image_encoder(image)["vision_features"]
                image_feature = image_feature.mean(dim=[2, 3])
            
            feats.append(image_feature.squeeze().detach().cpu().numpy())
            name_list.append(image_pth)

            self.save_feature(feats, name_list, model="sam2", model_type=model_type)

        return feats, name_list

    def extract_medsam_feature(
        self, model_type="medsam_vit_b"
    ) -> Tuple[ndarray, List[str]]:
        image_encoder, preprocess = load_sam_image_encoder(vit_type=model_type)
        image_encoder = image_encoder.cuda()
        resize_transform = ResizeLongestSide(image_encoder.img_size)
        for image_pth in tqdm(self.image_pth):
            feats = []
            name_list = []
            image = self.read_image(image_pth).squeeze()
            image = resize_transform.apply_image(image)
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            image = preprocess(image).cuda()
            with torch.no_grad():
                image_feature = image_encoder(image)
                image_feature = image_feature.mean(dim=[2, 3])

            feats.append(image_feature.squeeze().detach().cpu().numpy())
            name_list.append(image_pth)

            self.save_feature(feats, name_list, model="medsam", model_type=model_type)

        return feats, name_list

    def extract_clip_feature(self, model_type="ViT-B/32") -> Tuple[ndarray, List[str]]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_type, device=device)
        for image_pth in tqdm(self.image_pth):
            feats = []
            name_list = []
            image = self.read_image(image_pth).squeeze()
            if image.ndim == 3:
                image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
            image = preprocess(image).unsqueeze(0).cuda()
            with torch.no_grad():
                image_feature = model.encode_image(image)

            feats.append(image_feature.squeeze().detach().cpu().numpy())
            name_list.append(image_pth)
            self.save_feature(feats, name_list, model="clip", model_type=model_type)

        return feats, name_list

    def extract_medclip_feature(self, model_type="ViT") -> Tuple[ndarray, List[str]]:
        if model_type == "ResNet":
            model = MedCLIPVisionModel(
                medclip_checkpoint="./Foundation_models/checkpoints/medclip_resnet"
            ).cuda()
        elif model_type == "ViT":
            model = MedCLIPVisionModelViT(
                medclip_checkpoint="./Foundation_models/checkpoints/medclip_vit"
            ).cuda()
        processor = MedCLIPProcessor()

        for image_pth in tqdm(self.image_pth):
            feats = []
            name_list = []
            image = self.read_image(image_pth).squeeze()
            if image.ndim == 3:
                image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            image = Image.fromarray(image)

            image = (
                processor(images=image, return_tensors="pt")["pixel_values"]
                .float()
                .cuda()
            )
            with torch.no_grad():
                image_feature = model(image)

            feats.append(image_feature.squeeze().detach().cpu().numpy())
            name_list.append(image_pth)
            self.save_feature(feats, name_list, model="medclip", model_type=model_type)

        return feats, name_list

    def extract_dino_feature(self, model_type="ViT-B/16") -> Tuple[ndarray, List[str]]:
        if model_type == "ViT-B/16":
            model_name = "facebook/dino-vitb16"
        elif model_type == "ViT-S/16":
            model_name = "facebook/dino-vits16"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).cuda()
        model.eval()
        for image_pth in tqdm(self.image_pth):
            feats = []
            name_list = []
            image = self.read_image(image_pth).squeeze()
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            image = Image.fromarray(image)

            image = (
                processor(images=image, return_tensors="pt")["pixel_values"]
                .float()
                .cuda()
            )
            with torch.no_grad():
                image_feature = model(image).pooler_output

            feats.append(image_feature.squeeze().detach().cpu().numpy())
            name_list.append(image_pth)
            self.save_feature(feats, name_list, model="dino", model_type=model_type)

        return feats, name_list

    def extract_dino2_feature(self, model_type) -> Tuple[ndarray, List[str]]:
        if model_type == "ViT-B/14":
            model_name = "facebook/dinov2-base"
        elif model_type == "ViT-G/14":
            model_name = "facebook/dinov2-giant"
        # processor = AutoImageProcessor.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).cuda()
        model.eval()
        for image_pth in tqdm(self.image_pth):
            feats = []
            name_list = []
            image = self.read_image(image_pth).squeeze()
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            image = Image.fromarray(image)

            image = (
                processor(images=image, return_tensors="pt")["pixel_values"]
                .float()
                .cuda()
            )
            with torch.no_grad():
                image_feature = model(image).pooler_output

            feats.append(image_feature.squeeze().detach().cpu().numpy())
            name_list.append(image_pth)
            self.save_feature(feats, name_list, model="dino2", model_type=model_type)

        return feats, name_list

    def extract_resnet_feature(
        self, model_type="resnet18"
    ) -> Tuple[ndarray, List[str]]:
        """
        使用ResNet18在ImageNet1K上的预训练模型提取特征

        返回:
            Tuple[ndarray, List[str]]: 特征数组和对应的图像路径列表
        """
        # 加载预训练模型
        model = models.resnet18(pretrained=True)
        model = model.cuda()
        model.eval()  # 设置为评估模式

        # 定义预处理流程
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        feats = []
        name_list = []

        for image_pth in tqdm(self.image_pth):
            # 读取图像
            image = self.read_image(image_pth).squeeze()
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
            if image.shape[-1] == 1:
                image = np.concatenate([image, image, image], axis=-1)

            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            # 转换为PIL Image以便进行预处理
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # 预处理图像
            image_tensor = (
                preprocess(image).unsqueeze(0).cuda()
            )  # 添加batch维度并移到GPU

            # 提取特征
            with torch.no_grad():
                # 获取倒数第二层的输出（全局平均池化前的特征）
                features = model.conv1(image_tensor)
                features = model.bn1(features)
                features = model.relu(features)
                features = model.maxpool(features)

                features = model.layer1(features)
                features = model.layer2(features)
                features = model.layer3(features)
                features = model.layer4(features)

                # 全局平均池化
                features = model.avgpool(features)
                features = torch.flatten(features, 1)

            # 保存特征
            feats.append(features.squeeze().detach().cpu().numpy())
            name_list.append(image_pth)

            # 保存特征到文件
            self.save_feature(feats, name_list, model="resnet", model_type=model_type)

        return feats, name_list

    def extract_feature(self, model, model_type):
        method = getattr(self, f"extract_{model}_feature", None)
        if method is None:
            raise ValueError(f"Unsupported model: {model}")
        feats, name_list = method(model_type=model_type)

        return feats, name_list

    def save_feature(self, feats, name_list, model, model_type):
        feature_dir = self.base_dir.replace("dataset", "feature")
        model_type = model_type.replace("/", "_")
        save_dir = os.path.join(feature_dir, f"{model}_{model_type}")
        os.makedirs(save_dir, exist_ok=True)

        for feat, name in zip(feats, name_list):
            _, suffix = os.path.splitext(name)
            save_path = os.path.join(
                save_dir,
                os.path.basename(name).replace(
                    suffix, f"_{model}_{model_type}{suffix}"
                ),
            )
            np.save(save_path, feat)

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        You can use the following models and versions:
            resnet: [resnet18]
            clip: [RN50, RN101, RN50x4, RN5016, RN50x64, ViT-B/14, ViT-B/32, ViT-L/14, ViT-L/14@336px]
            medclip: [ViT, ResNet]
            sam: [vit_b, vit_h]
            medsam: [medsam_vit_b]
            sam2: [2b+, 2l, 2s, 2t, 2.1b+, 2.1l, 2.1s, 2.1t]
            dino: [ViT-B/16, ViT-S/16]
            dinov2: [ViT-B/14, ViT-G/14]
            LLaVA-v1.5 and LLaVA-v1.6 has the same image encoder as the ViT-L/14@336px version of clip.
            LLaVA-v1 has the same image encoder as the ViT-L/14 and ViT-L/14@336px version of clip.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/dataset/Spleen",
        help="path to the dataset",
    )
    parser.add_argument(
        "--model_version_type",
        type=str,
        default="clip_version_RN50x64",
        help="model name and version type, e.g., clip_version_RN50x64",
    )
    args = parser.parse_args()

    feature_extractor = Feature_extractor(base_dir=args.data_dir)
    model, model_type = args.model_version_type.split("_version_")
    feats, name_list = feature_extractor.extract_feature(
        model=model, model_type=model_type
    )
