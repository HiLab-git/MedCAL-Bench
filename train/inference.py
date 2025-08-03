import torch.nn as nn
from datetime import datetime
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from datasets.dataset import dataset_reader_cls
import sys
import argparse
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import os
import pandas as pd
import numpy as np
import re
from medpy.metric.binary import dc, hd95
from scipy.ndimage import zoom
import SimpleITK as sitk
import logging
import torchvision.transforms as transforms
from PIL import Image
import torch
from tqdm import tqdm
from skimage import measure


def largestConnectComponent_3d(binaryimg, ratio=3):
    label_image, num = measure.label(binaryimg, background=0, return_num=True)
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if num > 1:
        for region in measure.regionprops(label_image):
            if region.area < areas[-1] / ratio:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1], coordinates[2]] = 0
    label_image = label_image.astype(np.uint8)
    label_image[np.where(label_image > 0)] = 1

    return label_image


def cal_single_volume(pred, gt, cal_hd95=False):
    gt = gt > 0

    if np.any(pred):
        dice_score = dc(pred, gt)
        if cal_hd95:
            hd95_score = hd95(pred, gt)
        else:
            hd95_score = 404
    else:
        dice_score = 0
        hd95_score = 404

    return dice_score, hd95_score


def compute_miou_mdice(pred, mask):
    """
    Compute mIOU and mDICE for 2D segmentation predictions and masks.

    Args:
        pred (np.ndarray): 2D array of predicted labels (integers).
        mask (np.ndarray): 2D array of ground truth labels (integers).

    Returns:
        tuple: (mIOU, mDICE) as floats, representing mean IOU and mean DICE.
    """
    # Ensure inputs are 2D numpy arrays with integer values
    pred = np.asarray(pred, dtype=np.int32)
    mask = np.asarray(mask, dtype=np.int32)

    # Get unique classes from both pred and mask
    classes = np.unique(np.concatenate([pred.ravel(), mask.ravel()]))

    ious = []
    dices = []

    for cls in classes:
        # Binary masks for the current class
        pred_cls = pred == cls
        mask_cls = mask == cls

        # Compute intersection and union
        intersection = np.logical_and(pred_cls, mask_cls).sum()
        union = np.logical_or(pred_cls, mask_cls).sum()
        pred_sum = pred_cls.sum()
        mask_sum = mask_cls.sum()

        # Handle edge cases
        if union == 0:
            # If no pixels of this class in both pred and mask, IOU/DICE = 1
            ious.append(1.0)
            dices.append(1.0)
            continue

        # Compute IOU
        iou = intersection / union
        ious.append(iou)

        # Compute DICE
        dice = (
            (2 * intersection) / (pred_sum + mask_sum)
            if (pred_sum + mask_sum) > 0
            else 1.0
        )
        dices.append(dice)
        # print(cls, iou, dice)

    # Compute mean IOU and DICE
    miou = np.mean(ious)
    mdice = np.mean(dices)

    return miou, mdice


def inference(
    model,
    base_dir,
    stage,
    dataset,
    img_size,
    output_dir,
    cal_hd95=False,
    save_nii=False,
):
    """
    Perform inference by sorting slices based on volume_id and slice_id,
    combining slices into 3D volumes, and running inference on the 3D volumes.
    """
    model.eval()
    csv_path = os.path.join(base_dir, "data/dataset", dataset, "splits", stage + ".csv")
    df = pd.read_csv(csv_path)

    # Extract volume_id and slice_id from file names
    df["volume_id"] = df["image_pth"].apply(
        lambda x: os.path.basename(x).split("_slice_")[0]
    )
    df["slice_id"] = df["image_pth"].apply(
        lambda x: int(re.search(r"_slice_(\d+)", os.path.basename(x)).group(1))
    )

    # Sort by volume_id and slice_id
    df = df.sort_values(by=["volume_id", "slice_id"])

    grouped = df.groupby("volume_id")

    all_dice_scores = []
    all_hd95_scores = []

    for volume_id, group in grouped:
        slices = []
        masks = []

        for _, row in group.iterrows():

            image = np.load(row["image_pth"])
            mask = np.load(row["mask_pth"])

            # Resize image and mask to the target size
            # print(image.shape)
            image = (image - image.mean()) / image.std()
            # print(image.shape)
            image = image.reshape(1, image.shape[0], image.shape[1])
            # image = np.transpose(image, (2, 0, 1))
            # zoom_factors = (1.0, img_size / image.shape[1], img_size / image.shape[2])
            # image_resized = zoom(image, zoom_factors, order=3)  # Bilinear interpolation
            image_resized = image

            slices.append(image_resized)
            masks.append(mask)

        # Combine slices into a 3D volume
        slices = np.array(slices)
        masks = np.array(masks)

        preds = []
        for i in tqdm(range(slices.shape[0]), desc=f"Inferencing {volume_id}"):
            slice_input = (
                torch.from_numpy(slices[i]).unsqueeze(0).float().cuda()
            )  # Add batch and channel dimensions

            pred = (
                model(slice_input)
                .argmax(dim=1)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            pred = zoom(
                pred, (masks.shape[1] / img_size, masks.shape[2] / img_size), order=0
            )
            preds.append(pred)

        preds = np.array(preds)

        preds = largestConnectComponent_3d(preds)
        masks[masks != 1] = 0

        dice_score, hd95_score = cal_single_volume(preds, masks, cal_hd95)
        all_dice_scores.append(dice_score)
        all_hd95_scores.append(hd95_score)

        logging.info(
            f"Volume {volume_id}: Dice = {dice_score:.4f}, HD95 = {hd95_score:.4f}"
        )

        # Save predictions as .nii file if save_nii is True
        if save_nii:
            os.makedirs(os.path.join(output_dir, stage), exist_ok=True)
            nii_path = os.path.join(output_dir, stage, f"{volume_id}_prediction.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(preds), nii_path)

    if save_nii:
        logging.info(f"Predictions are saved to {os.path.join(output_dir, stage)}")

    all_dice_scores = np.array(all_dice_scores)
    all_hd95_scores = np.array(all_hd95_scores)
    model.train()

    return (
        all_dice_scores.mean(),
        all_dice_scores.std(),
        all_hd95_scores.mean(),
        all_hd95_scores.std(),
    )


def inference_2d(
    model, base_dir, stage, dataset, img_size, output_dir, save_pred=False
):
    model.eval()
    csv_path = os.path.join(base_dir, "data/dataset", dataset, "splits", stage + ".csv")
    df = pd.read_csv(csv_path)
    image_pth = df["image_pth"].tolist()
    mask_pth = df["mask_pth"].tolist()

    all_miou_scores = []
    all_mdice_scores = []

    for img_pth, m_pth in zip(image_pth, mask_pth):
        image = np.load(img_pth)
        mask = np.load(m_pth)
        image = (image - image.mean()) / image.std()
        image = np.transpose(image, (2, 0, 1))
        zoom_factors = (1.0, img_size / image.shape[1], img_size / image.shape[2])
        image_resized = zoom(image, zoom_factors, order=3)
        slice_input = (
            torch.from_numpy(image_resized).unsqueeze(0).float().cuda()
        )  # Add batch and channel dimensions
        pred = model(slice_input).argmax(dim=1).squeeze().detach().cpu().numpy()
        pred = zoom(pred, (mask.shape[0] / img_size, mask.shape[1] / img_size), order=0)

        miou, mdice = compute_miou_mdice(pred, mask)

        all_miou_scores.append(miou)
        all_mdice_scores.append(mdice)

        logging.info(
            f"Slice {os.path.basename(img_pth)}: mIoU = {miou:.4f}, mDice = {mdice:.4f}"
        )

        if save_pred:
            os.makedirs(os.path.join(output_dir, stage), exist_ok=True)
            pred_path = os.path.join(
                output_dir, stage, f"{os.path.basename.split('.')[0]}_prediction.png"
            )
            pred = (pred * 255).astype(np.uint8)
            pred.save(pred_path)

    if save_pred:
        logging.info(f"Predictions are saved to {os.path.join(output_dir, stage)}")

    all_mdice_scores = np.array(all_mdice_scores)
    all_miou_scores = np.array(all_miou_scores)
    model.train()

    return (
        all_mdice_scores.mean(),
        all_mdice_scores.std(),
        all_miou_scores.mean(),
        all_miou_scores.std(),
    )


def inference_2d_Kvasir(
    model, base_dir, stage, dataset, img_size, output_dir, save_pred=False
):
    model.eval()
    all_dices = []
    csv_paths = [
        os.path.join(base_dir, "data/dataset", dataset, "splits", stage + ".csv")
    ]
    targets = ["training"] if stage == "train" else ["test"]
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        image_pth = df["image_pth"].tolist()
        mask_pth = df["mask_pth"].tolist()

        all_dice = []

        for img_pth, m_pth in tqdm(
            zip(image_pth, mask_pth), desc=f"Inferencing {targets[i]}"
        ):
            image = np.load(img_pth)
            mask = np.load(m_pth).squeeze(0)
            mask[mask == -1] = 0

            image = (image - image.mean()) / image.std()
            image = np.transpose(image, (2, 0, 1))
            zoom_factors = (1.0, img_size / image.shape[1], img_size / image.shape[2])
            image_resized = zoom(image, zoom_factors, order=3)
            slice_input = (
                torch.from_numpy(image_resized).unsqueeze(0).float().cuda()
            )  # Add batch and channel dimensions
            pred = model(slice_input).argmax(dim=1).squeeze().detach().cpu().numpy()
            print(pred.shape, mask.shape)
            pred = zoom(
                pred, (mask.shape[0] / img_size, mask.shape[1] / img_size), order=0
            )

            dice = dc(pred, mask)

            all_dice.append(dice)
            if save_pred:
                if stage == "train":
                    save_folder = os.path.join(output_dir, f"visual_train")
                else:
                    save_folder = os.path.join(output_dir, f"visual_{targets[i]}")
                os.makedirs(save_folder, exist_ok=True)
                Image.fromarray(np.squeeze(mask * 255).astype(np.uint8)).save(
                    os.path.join(
                        save_folder,
                        os.path.basename(img_pth).replace(".npy", "_mask.png"),
                    )
                )
                Image.fromarray(np.squeeze(pred * 255).astype(np.uint8)).save(
                    os.path.join(
                        save_folder,
                        os.path.basename(img_pth).replace(".npy", "_pred.png"),
                    )
                )
                image = (image - image.min()) / (image.max() - image.min()) * 255
                image = image.astype(np.uint8)
                image = np.transpose(image, (1, 2, 0))
                Image.fromarray(np.squeeze(image)).save(
                    os.path.join(
                        save_folder,
                        os.path.basename(img_pth).replace(".npy", "_img.png"),
                    )
                )

            logging.info(f"Slice {os.path.basename(img_pth)}: Dice = {dice:.4f}")

        all_dice = np.array(all_dice)
        all_dices.append(all_dice.mean())

    for dice, target in zip(all_dices, targets):
        logging.info(f"Dataset {target}: Dice = {dice:.4f}")

    all_dice = np.array(all_dice)
    logging.info(f"Mean Dice = {all_dice.mean():.4f}±{all_dice.std()}")
    model.train()

    return all_dice.mean(), all_dice.std(), None, None


def inference_2d_Polyp(
    model, base_dir, stage, dataset, img_size, output_dir, save_pred=False
):
    model.eval()
    all_dices = []
    # csv_paths = ['/media/ubuntu/FM-AL/data/dataset/Polyp_preprocessed/splits/train.csv']
    if stage == "train":
        csv_paths = [
            os.path.join(base_dir, "data/dataset", dataset, "splits", stage + ".csv")
        ]
        targets = ["training set"]
    else:
        targets = [
            "CVC-300",
            "CVC-ClinicDB",
            "CVC-ColonDB",
            "ETIS-LaribPolypDB",
            "Kvasir",
        ]
        csv_paths = [
            os.path.join(
                base_dir, "data/dataset", dataset, "splits", stage + f"_{target}" ".csv"
            )
            for target in targets
        ]
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        image_pth = df["image_pth"].tolist()
        mask_pth = df["mask_pth"].tolist()

        all_dice = []

        for img_pth, m_pth in tqdm(
            zip(image_pth, mask_pth), desc=f"Inferencing {targets[i]}"
        ):
            image = np.load(img_pth)
            mask = np.load(m_pth).squeeze(0)
            mask[mask == -1] = 0

            image = (image - image.mean()) / image.std()
            image = np.transpose(image, (2, 0, 1))
            zoom_factors = (1.0, img_size / image.shape[1], img_size / image.shape[2])
            image_resized = zoom(image, zoom_factors, order=3)
            slice_input = (
                torch.from_numpy(image_resized).unsqueeze(0).float().cuda()
            )  # Add batch and channel dimensions
            pred = model(slice_input).argmax(dim=1).squeeze().detach().cpu().numpy()
            pred = zoom(
                pred, (mask.shape[0] / img_size, mask.shape[1] / img_size), order=0
            )

            dice = dc(pred, mask)

            all_dice.append(dice)
            if save_pred:
                if stage == "train":
                    save_folder = os.path.join(output_dir, f"visual_train")
                else:
                    save_folder = os.path.join(output_dir, f"visual_{targets[i]}")
                os.makedirs(save_folder, exist_ok=True)
                Image.fromarray(np.squeeze(mask * 255).astype(np.uint8)).save(
                    os.path.join(
                        save_folder,
                        os.path.basename(img_pth).replace(".npy", "_mask.png"),
                    )
                )
                Image.fromarray(np.squeeze(pred * 255).astype(np.uint8)).save(
                    os.path.join(
                        save_folder,
                        os.path.basename(img_pth).replace(".npy", "_pred.png"),
                    )
                )
                image = (image - image.min()) / (image.max() - image.min()) * 255
                image = image.astype(np.uint8)
                image = np.transpose(image, (1, 2, 0))
                Image.fromarray(np.squeeze(image)).save(
                    os.path.join(
                        save_folder,
                        os.path.basename(img_pth).replace(".npy", "_img.png"),
                    )
                )

            logging.info(f"Slice {os.path.basename(img_pth)}: Dice = {dice:.4f}")

        all_dice = np.array(all_dice)
        all_dices.append(all_dice.mean())

    for dice, target in zip(all_dices, targets):
        logging.info(f"Dataset {target}: Dice = {dice:.4f}")

    all_dice = np.array(all_dice)
    logging.info(f"Mean Dice = {all_dice.mean():.4f}±{all_dice.std()}")
    model.train()

    return all_dice.mean(), all_dice.std(), None, None


def inference_2d_TN3K(
    model, base_dir, stage, dataset, img_size, output_dir, save_pred=False
):
    model.eval()
    csv_path = os.path.join(base_dir, "data/dataset", dataset, "splits", stage + ".csv")
    df = pd.read_csv(csv_path)
    image_pth = df["image_pth"].tolist()
    mask_pth = df["mask_pth"].tolist()

    all_dice = []

    for img_pth, m_pth in tqdm(
        zip(image_pth, mask_pth), desc=f"Inferencing {stage} set"
    ):
        image = np.load(img_pth)
        mask = np.load(m_pth).squeeze(2)
        mask[mask == -1] = 0

        image = (image - image.mean()) / image.std()
        image = np.transpose(image, (2, 0, 1))
        zoom_factors = (1.0, img_size / image.shape[1], img_size / image.shape[2])
        image_resized = zoom(image, zoom_factors, order=3)
        slice_input = (
            torch.from_numpy(image_resized).unsqueeze(0).float().cuda()
        )  # Add batch and channel dimensions
        pred = model(slice_input).argmax(dim=1).squeeze().detach().cpu().numpy()
        pred = zoom(pred, (mask.shape[0] / img_size, mask.shape[1] / img_size), order=0)

        dice = dc(pred, mask)

        all_dice.append(dice)
        if save_pred:
            save_folder = os.path.join(output_dir, f"visual_{stage}")
            os.makedirs(save_folder, exist_ok=True)
            Image.fromarray(np.squeeze(mask * 255).astype(np.uint8)).save(
                os.path.join(
                    save_folder, os.path.basename(img_pth).replace(".npy", "_mask.png")
                )
            )
            Image.fromarray(np.squeeze(pred * 255).astype(np.uint8)).save(
                os.path.join(
                    save_folder, os.path.basename(img_pth).replace(".npy", "_pred.png")
                )
            )
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
            Image.fromarray(np.squeeze(image)).save(
                os.path.join(
                    save_folder, os.path.basename(img_pth).replace(".npy", "_img.png")
                )
            )

        logging.info(f"Slice {os.path.basename(img_pth)}: Dice = {dice:.4f}")

    all_dice = np.array(all_dice)
    logging.info(f"Mean Dice: {all_dice.mean():.4f}, Std: {all_dice.std():.4f}")

    model.train()

    return all_dice.mean(), all_dice.std(), None, None


def inference_2d_cls(model, base_dir, stage, dataset):
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model.eval()
    correct = 0
    total = 0
    csv_path = os.path.join(base_dir, "data/dataset", dataset, "splits", stage + ".csv")
    test_dataset = dataset_reader_cls(csv_path=csv_path, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    with torch.no_grad():
        for samples in test_loader:
            inputs, labels = samples["image"], samples["label"]
            inputs, labels = inputs.cuda(), labels.cuda()
            labels = labels.squeeze(1)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--adpt_ckpt_dir",
        type=str,
        default="../train/outputs/Heart/clip_RN50x64/ALPS_109",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "-b", "--base_dir", type=str, default="../../", help="base directory of FM-AL"
    )
    parser.add_argument(
        "-s", "--stage", type=str, default="train", help="inference stage"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=[
            "Heart_preprocessed",
            "Spleen_preprocessed",
            "Spleen_Date512new",
            "Kvasir_preprocessed",
            "TN3K_preprocessed",
            "Derma",
            "Pneumonia",
            "Breast",
        ],
        default="Heart",
        help="dataset name",
    )
    parser.add_argument(
        "-n", "--num_classes", type=int, default=2, help="number of classes"
    )
    parser.add_argument("--img_size", type=int, default=256, help="image size")
    parser.add_argument(
        "--input_nc", type=int, default=1, help="number of input channels"
    )
    parser.add_argument(
        "--save_pred",
        action="store_true",
        help="whether to save predictions as .nii file",
    )
    args = parser.parse_args()
    if args.dataset in ["Derma", "Pneumonia", "Breast"]:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    else:
        model = UNet(
            spatial_dims=2,
            in_channels=args.input_nc,
            out_channels=args.num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
    ckpt_path = os.path.join(args.adpt_ckpt_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.adpt_ckpt_dir, "final_model.pth")
    model.load_state_dict(torch.load(ckpt_path))
    # model.load_state_dict(torch.load(os.path.join(args.adpt_ckpt_dir, "best_model.pth")))
    # model.load_state_dict(torch.load(os.path.join(args.adpt_ckpt_dir, "/media/ubuntu/FM-AL/train/outputs/Spleen_preprocessed/splits/train/epoch_460_0.9334.pth")))
    model = model.cuda()
    output_filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(args.adpt_ckpt_dir, "testing_log"), exist_ok=True)

    print(os.path.join(args.adpt_ckpt_dir, "testing_log", output_filename + ".log"))

    logging.basicConfig(
        filename=os.path.join(
            args.adpt_ckpt_dir, "testing_log", output_filename + ".log"
        ),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.dataset == "Kvasir_preprocessed":
        dc_mean, dc_std, hd95_mean, hd95_std = inference_2d_Kvasir(
            model=model,
            base_dir=args.base_dir,
            stage=args.stage,
            dataset=args.dataset,
            img_size=args.img_size,
            output_dir=args.adpt_ckpt_dir,
            save_pred=args.save_pred,
        )

    elif args.dataset == "TN3K_preprocessed":
        dc_mean, dc_std, hd95_mean, hd95_std = inference_2d_TN3K(
            model=model,
            base_dir=args.base_dir,
            stage=args.stage,
            dataset=args.dataset,
            img_size=args.img_size,
            output_dir=args.adpt_ckpt_dir,
            save_pred=args.save_pred,
        )

    elif args.dataset in ["Derma", "Pneumonia", "Breast"]:
        acc_mean = inference_2d_cls(
            model=model,
            base_dir=args.base_dir,
            stage=args.stage,
            dataset=args.dataset,
        )

        logging.info(f"Accuracy mean: {acc_mean:.4f}")
    else:
        dc_mean, dc_std, hd95_mean, hd95_std = inference(
            model=model,
            base_dir=args.base_dir,
            stage=args.stage,
            dataset=args.dataset,
            img_size=args.img_size,
            output_dir=args.adpt_ckpt_dir,
            cal_hd95=True,
            save_nii=args.save_pred,
        )

        logging.info(f"Dice mean: {dc_mean:.4f}, Dice std: {dc_std:.4f}")
        logging.info(f"HD95 mean: {hd95_mean:.4f}, HD95 std: {hd95_std:.4f}")
        logging.info("Inference Finished!")
