from PIL import Image
import torch
import logging
import random
from datasets.dataset import dataset_reader, RandomGenerator, RandomGenerator_Simple
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from monai.networks.utils import one_hot
from tqdm import tqdm
from datetime import datetime
import os
import sys
from utils.losses import DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from inference import inference, inference_2d_Kvasir, inference_2d_TN3K
from datasets.transforms import *


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = UNet(
            spatial_dims=args.dim,
            in_channels=args.input_nc,
            out_channels=args.num_classes,
            # channels=(16, 32, 48, 64, 128, 128, 256),
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=args.dropout,
        )
        self.config_path = "/media/ubuntu/FM-AL/data/dataset/config/" + args.dataset + ".json"
        self.train_transform, self.val_transform = Transform_generator(
            config_path=self.config_path
        ).tr_val_trans()
        db_train = dataset_reader(
            csv_path=args.plan_path,
            # transform=transforms.Compose(
            #     [
            #         RandomGenerator_Simple(
            #             output_size=[args.img_size, args.img_size],
            #         ),
            #     ]
            # ),
            transform=self.train_transform,
        )
        

        def worker_init_fn(worker_id):
            random.seed(args.seed + worker_id)

        self.train_loader = DataLoader(
            db_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        
        self.Dice_Loss = DiceLoss(n_classes=args.num_classes)
        # self.loss_fn = DiceCELoss(to_onehot_y=False, softmax=True).cuda()
        self.CrossEntropy_Loss = CrossEntropyLoss()
        self.optimizer = optim.SGD(
            params=self.model.parameters(),
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

    def cal_loss(self, outputs, label_batch):
        ce_loss = self.CrossEntropy_Loss(outputs, label_batch.squeeze(1))
        dice_loss = self.Dice_Loss(outputs, label_batch, softmax=True)
        total_loss = 0.5 * (ce_loss + dice_loss)

        return total_loss, ce_loss, dice_loss
    
    def prep_label(self, target):
        """Manually convert to one-hot based on the number of classes
        This is useful when foreground labels may be absent for certain subjects
        e.g., male subjects with removed prostate; prostate channel will be all zeros 
        """
        return one_hot(target, self.args.num_classes, dim=1)

    def train(self):
        args = self.args
        model = self.model.cuda()
        trainloader = self.train_loader
        optimizer = self.optimizer
        max_epochs = args.max_epochs
        base_dir = args.base_dir
        base_lr = args.base_lr
        val_epochs = args.val_epochs

        output_filename = datetime.now().strftime("%Y%m%d-%H%M%S")

        output_dir = os.path.join(
            base_dir,
            "train/outputs",
            args.dataset,
            args.plan_path.split("/")[-2],
            args.plan_path.split("/")[-1].split(".")[0],
        )

        os.makedirs(os.path.join(output_dir, "log"), exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(output_dir, "log", output_filename + ".log"),
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        model.train()

        iter_num = 0
        max_epoch = max_epochs
        max_iterations = args.max_iter
        logging.info(
            "{} iterations per epoch. {} max iterations ".format(
                len(trainloader), max_iterations
            )
        )
        best_performance = 0.0

        iterator = tqdm(range(max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(trainloader):
                image_batch, label_batch = (
                    sampled_batch["image"].float().cuda(),
                    sampled_batch["segmentation"],
                )
                case_name = sampled_batch["case_name"]

                outputs = model(image_batch) # [B, 2, H, W]
                
                               
                # label_batch = self.prep_label(label_batch) # [B, C, H, W]
         
                loss, loss_ce, loss_dice = self.cal_loss(outputs, label_batch[0].long().cuda())
                # loss, loss_ce, loss_dice = self.cal_loss(outputs, label_batch.long().cuda())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_

                iter_num = iter_num + 1

                logging.info(
                    "iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f"
                    % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_)
                )
               
                """ check figure and save """ 
                # if iter_num > 10000 and (abs(loss_dice.item() - 0.5) < 0.1):
                #     case_name = case_name[0]
                #     os.makedirs(os.path.join(output_dir, "saved_img"), exist_ok=True)
                #     image = image_batch[0].squeeze().detach().cpu().numpy()
                #     image = (image - image.min()) / (image.max() - image.min()) * 255
                #     image = image.astype(np.uint8)
                #     outputs_arg = torch.argmax(outputs, dim=1)[0]
                #     output_np = outputs_arg.squeeze().detach().cpu().numpy() 
                #     output_pred = (output_np * 255).astype(np.uint8)  
                #     save_path = os.path.join(output_dir, "saved_img", f"case{os.path.basename(case_name).split('.')[0]}_pred.png")
                #     save_path2 = os.path.join(output_dir, "saved_img", f"case{os.path.basename(case_name).split('.')[0]}_mask.png")
                #     save_path3 = os.path.join(output_dir, "saved_img", f"case{os.path.basename(case_name).split('.')[0]}.png")
                #     image = Image.fromarray(image)
                #     output_pred = Image.fromarray(output_pred)
                #     output_mask = Image.fromarray((label_batch[0][0].squeeze().detach().cpu().numpy() * 255).astype(np.uint8))
                
                #     image.save(save_path3)
                #     output_pred.save(save_path)
                #     output_mask.save(save_path2)
                
                if iter_num > args.max_iter:
                    break

            if (epoch_num + 1) % val_epochs == 0:
                if args.dataset == 'Kvasir_preprocessed':
                    dice_mean, dice_std, _, _ = inference_2d_Kvasir(
                        model=model,
                        base_dir=base_dir,
                        stage="test",
                        dataset=args.dataset,
                        img_size=args.img_size,
                        output_dir=output_dir,
                        save_pred=False,
                    )
                elif args.dataset == "TN3K_preprocessed":
                    dice_mean, dice_std, _, _ = inference_2d_TN3K(
                        model=model,
                        base_dir=base_dir,
                        stage="valid",
                        dataset=args.dataset,
                        img_size=args.img_size,
                        output_dir=output_dir,
                        save_pred=False,
                    )
                else:
                    dice_mean, dice_std, _, _ = inference(
                        model=model,
                        base_dir=base_dir,
                        stage="valid",
                        dataset=args.dataset,
                        img_size=args.img_size,
                        output_dir=output_dir,
                        cal_hd95=False,
                        save_nii=False,
                    )

                if dice_mean > best_performance:
                    best_performance = dice_mean

                    save_mode_path = os.path.join(
                        output_dir, f"epoch_{str(epoch_num + 1)}_{best_performance:.4f}.pth"
                    )
                    save_best_mode_path = os.path.join(
                        output_dir, f"best_model.pth"
                    )
                    
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_mode_path)

                save_final_mode_path = os.path.join(
                    output_dir, f"final_model.pth"
                )
                torch.save(model.state_dict(), save_final_mode_path)

            
                

            if iter_num > args.max_iter:
                break
                

        iterator.close()

        return "Training Finished!"
