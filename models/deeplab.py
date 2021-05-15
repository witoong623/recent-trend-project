import segmentation_models_pytorch as smp
from utils import no_pretrain_precessing


def get_model(num_classes, encoder='resnet50', encoder_weight='imagenet', activation='sigmoid'):
    ''' return model and optional preprocessing function '''

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=encoder, 
        encoder_weights=None, 
        classes=num_classes, 
        activation=activation,
    )

    if encoder_weight is None:
        preprocessing_fn = no_pretrain_precessing
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weight)

    return model, preprocessing_fn
