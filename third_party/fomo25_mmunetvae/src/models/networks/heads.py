import torch
import torch.nn as nn


class ClsRegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Just linear
        # self.fc = nn.Linear(in_channels, num_classes)

        # Simple non-linearity
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            # nn.ReLU(inplace=True),
            nn.SiLU(),
            # nn.Linear(in_channels // 2, in_channels // 2),
            # nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_channels // 2, num_classes)
        )

    def forward(self, x):
        # x = x[-1]  # only use bottleneck repr
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
