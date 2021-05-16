import argparse
import copy
import dataset_utils
import os
import torch
import segmentation_models_pytorch as smp
from config import cfg
from datasets import LandCoverDataset
from models.deeplab import get_model as get_deeplab_model
from torch.utils.data import DataLoader, Subset
from utils import get_validation_augmentation, get_training_augmentation, get_preprocessing, save_history, save_model
from train_utils import TrainEpochMultiGPU, ValidEpochMultiGPU

LANDCOVER_ROOT = '/root/deepglobe'

def training(model, train_loader, val_loader, cfg):
    if len(cfg.TRAIN.multi_gpus) > 1:
        DEVICE = cfg.TRAIN.multi_gpus
    else:
        DEVICE = torch.device(f'cuda:{cfg.TRAIN.gpu_id}' if torch.cuda.is_available() else "cpu")

    # index (or channel) 6 in ground truth is unknow
    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # define optimizer
    if cfg.TRAIN.optimizer == 'sgd':
        optimizer = torch.optim.SGD([ 
            dict(params=model.parameters(), lr=cfg.TRAIN.learning_rate, momentum=cfg.TRAIN.momentum)
        ])
    elif cfg.TRAIN.optimizer == 'adam':
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=cfg.TRAIN.learning_rate)
        ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, step_size=5, gamma=0.8)


    # load best saved model checkpoint from previous commit (if present)
    # if os.path.exists('../input/deepglobe-land-cover-classification-deeplabv3/best_model.pth'):
    #     model = torch.load('../input/deepglobe-land-cover-classification-deeplabv3/best_model.pth', map_location=DEVICE)
    #     print('Loaded pre-trained DeepLabV3+ model!')

    train_epoch = TrainEpochMultiGPU(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpochMultiGPU(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []
    model_name = cfg.MODEL.name

    for i in range(0, cfg.TRAIN.epochs):
        # Perform training & validation
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        lr_scheduler.step(i + 1)

        save_history((cfg.TRAIN, train_logs_list), f'{model_name}_epoch{i+1}_train', os.path.join(cfg.history_dir, model_name))
        save_history((cfg.VAL, valid_logs_list), f'{model_name}_epoch{i+1}_val', os.path.join(cfg.history_dir, model_name))

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']

            save_model(model, 'best_model', os.path.join(cfg.weight_dir, cfg.MODEL.name))

    print('train finished')


if __name__ == '__main__':
    # config prep
    parser = argparse.ArgumentParser(
        description="RTML deep globe traning"
    )

    parser.add_argument(
        "--cfg",
        default="cfg/deeplab_resnet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    print(cfg)

    # get information about dataset
    train_df, val_df = dataset_utils.get_landcover_train_val_df(LANDCOVER_ROOT, random_state=cfg.SEED)
    dataset_info = dataset_utils.get_landcover_info(LANDCOVER_ROOT, include_unknow=False)
    class_names = dataset_info['class_names']
    class_rgb_values = dataset_info['class_rgb_values']
    select_class_rgb_values = dataset_info['select_class_rgb_values']

    num_classes = len(select_class_rgb_values)

    model, preprocessing_fn = get_deeplab_model(num_classes, cfg.MODEL.encoder)

    # Get train and val dataset instances
    train_dataset = LandCoverDataset(
        train_df, 
        augmentation=get_training_augmentation(cfg.TRAIN.augment_type),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    valid_dataset = LandCoverDataset(
        val_df, 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    if cfg.DEBUG:
        # if I only want to debug code, train and val only for 10 samples
        train_dataset = Subset(train_dataset, [n for n in range(10)])
        valid_dataset = Subset(valid_dataset, [n for n in range(10)])

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=cfg.TRAIN.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=cfg.VAL.batch_size, shuffle=False, pin_memory=True, num_workers=cfg.VAL.num_workers)

    training(model, train_loader, val_loader, cfg)
