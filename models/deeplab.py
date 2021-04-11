import segmentation_models_pytorch as smp


def get_model(num_classes, encoder='resnet50', encoder_weight='imagenet', activation='sigmoid'):
    ''' return model and optional preprocessing function '''

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=encoder, 
        encoder_weights=encoder_weight, 
        classes=num_classes, 
        activation=activation,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weight)

    return model, preprocessing_fn
