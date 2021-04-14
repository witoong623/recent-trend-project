from yacs.config import CfgNode as CN


cfg = CN()

# config
cfg.DIR = 'cfg/deeplab_resnet.yaml'
cfg.SEED = 42
cfg.DEBUG = False

# ----- model -----
cfg.MODEL = CN()
# must be valid name of derectory too
cfg.MODEL.name = 'deeplabv3_resnet'

# ----- training -----
cfg.TRAIN = CN()
cfg.TRAIN.epochs = 20
cfg.TRAIN.learning_rate = 0.01
cfg.TRAIN.momentum = 0
cfg.TRAIN.batch_size = 2
cfg.TRAIN.num_workers = 1
cfg.TRAIN.gpu_id = 0
cfg.TRAIN.history_dir = 'history/train'

# ----- validation -----
cfg.VAL = CN()
cfg.VAL.batch_size = 2
cfg.VAL.num_workers = 1
cfg.VAL.history_dir = 'history/validate'

# ----- testing -----
