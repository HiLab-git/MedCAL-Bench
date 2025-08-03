import argparse
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
from datetime import datetime
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.dataset import dataset_reader_cls
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import argparse


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sample in tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{num_epochs}]"):
        inputs, labels = sample["image"], sample["label"]
        inputs, labels = inputs.cuda(), labels.cuda()
        labels = labels.squeeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    return running_loss / len(train_loader), 100.0 * correct / total


# 验证
def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for samples in val_loader:
            inputs, labels = samples['image'], samples['label']
            inputs, labels = inputs.cuda(), labels.cuda()
            labels = labels.squeeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), 100.0 * correct / total


# 主训练函数
def train_model(args):
    # 加载数据
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = dataset_reader_cls(
        csv_path=args.plan_path, transform=train_transform
    )
    val_dataset = dataset_reader_cls(
        csv_path='/media/ubuntu/FM-AL/data/dataset/'+ args.dataset + '/splits/valid.csv', transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    num_classes = args.num_classes
    num_epochs = args.num_epochs
    base_dir = args.base_dir

    output_dir = os.path.join(
        base_dir,
        "train/outputs",
        args.dataset,
        args.plan_path.split("/")[-2],
        args.plan_path.split("/")[-1].split(".")[0],
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(output_dir, "log"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "log", output_filename + ".log"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.cuda()
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,  # 初始学习率
        betas=(0.9, 0.999),
        weight_decay=0.05,  # L2正则化
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 初始周期
        T_mult=2,  # 周期倍增因子
        eta_min=1e-6,  # 最小学习率
    )

    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, num_epochs
        )

        logging.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        if (epoch + 1) % 5 == 0:  # 每10个epoch验证一次
            val_loss, val_acc = validate(model, val_loader, criterion)
            logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if val_acc > best_acc:
                logging.info(f"New best model found at epoch {epoch + 1} with accuracy {val_acc:.4f}%")
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"epoch_{epoch + 1}_{best_acc:.4f}.pth"),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model")
    parser.add_argument(
        "--plan_path",
        type=str,
        default="../data/dataset/Derma/splits/train.csv",
        help="path of the plan file for customized selection",
    )
    parser.add_argument(
        "--base_dir", type=str, default="..", help="base directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2000, help="Number of epochs to train"
    )
    
    parser.add_argument(
        "--img_size", type=int, default=224, help="Input image size for the model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Derma",
        help="Name of the dataset being used",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of classes in the dataset",
    )

    parser.add_argument('--deterministic', type=int, default=1, help="whether use deterministic training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Experiment Parameters
    train_model(args)
