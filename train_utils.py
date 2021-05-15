import torch
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch


class TrainEpochMultiGPU(TrainEpoch):
    def __init__(self, model, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def _to_device(self):
        device = self.device
        if isinstance(self.device, list) and len(self.device) > 1:
            if not isinstance(self.model, torch.nn.DataParallel):
                self.model = torch.nn.DataParallel(self.model, self.device)

            self.device = f'cuda:{self.device[0]}'
        elif isinstance(self.device, list):
            raise ValueError('list of device must be greter than 1, otherwise provide a device in string format')

        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)


class ValidEpochMultiGPU(ValidEpoch):
    def __init__(self, model, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def _to_device(self):
        device = self.device
        if isinstance(self.device, list) and len(self.device) > 1:
            if not isinstance(self.model, torch.nn.DataParallel):
                self.model = torch.nn.DataParallel(self.model, self.device)

            self.device = f'cuda:{self.device[0]}'
        elif isinstance(self.device, list):
            raise ValueError('list of device must be greter than 1, otherwise provide a device in string format')

        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)
