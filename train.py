import dataset_utils
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from datasets import LandCoverDataset
from utils import get_validation_augmentation, get_training_augmentation, get_preprocessing
from models.deeplab import get_model as get_deeplab_model

LANDCOVER_ROOT = '/root/deepglobe'

def training(model, train_loader, val_loader):
    epoch = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.00008),
    ])

    # define learning rate scheduler (not used in this NB)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    # )

    # load best saved model checkpoint from previous commit (if present)
    # if os.path.exists('../input/deepglobe-land-cover-classification-deeplabv3/best_model.pth'):
    #     model = torch.load('../input/deepglobe-land-cover-classification-deeplabv3/best_model.pth', map_location=DEVICE)
    #     print('Loaded pre-trained DeepLabV3+ model!')

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, epoch):
        # Perform training & validation
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

    print(f'train finished')


if __name__ == '__main__':
    # get information about dataset
    train_df, val_df = dataset_utils.get_landcover_train_val_df(LANDCOVER_ROOT, random_state=42)
    dataset_info = dataset_utils.get_landcover_info(LANDCOVER_ROOT)
    class_names = dataset_info['class_names']
    class_rgb_values = dataset_info['class_rgb_values']
    select_class_rgb_values = dataset_info['select_class_rgb_values']

    num_classes = len(select_class_rgb_values)

    model, preprocessing_fn = get_deeplab_model(num_classes)

    # Get train and val dataset instances
    train_dataset = LandCoverDataset(
        train_df, 
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    valid_dataset = LandCoverDataset(
        val_df, 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    training(model, train_loader, val_loader)
