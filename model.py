import torch.nn as nn
import torchvision


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.before_model = torchvision.models.efficientnet_b0(num_classes=1)
        self.after_model = torchvision.models.efficientnet_b0(num_classes=1)

    def forward(self, before, after):
        before = self.before_model(before)
        after = self.after_model(after)

        return before - after
