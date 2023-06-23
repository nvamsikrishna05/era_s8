import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_SIZE = 2

def get_normalization(type, output_size, group_size):
    if type == 'bn':
        return nn.BatchNorm2d(output_size)
    elif type == 'gn':
        return nn.GroupNorm(group_size, output_size)
    elif type == 'ln':
        return nn.GroupNorm(1, output_size)

class NetInitial(nn.Module):
    def __init__(self, normalization) -> None:
        super(NetInitial, self).__init__()
        dropout_value = 0.1
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 32

        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 64 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 32

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
        ) # 32

        self.P1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16

        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 64 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 16

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 128 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 16

        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 256 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 16

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, padding=0),
        ) # 16

        self.P2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 64 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 8

        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 128 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 8

        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 256 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 6

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # Block 1
        x = self.C1(x)
        x = self.C2(x)
        
        x = self.c1(x)
        x = self.P1(x)
        
        # Block 2
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)

        x = self.c2(x)
        x = self.P2(x)

        # Block 3
        x = self.C6(x)
        x = self.C7(x)
        x = self.C8(x)

        x= self.gap(x)
        x = self.c3(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class NetInitial2(nn.Module):
    def __init__(self) -> None:
        super(NetInitial2, self).__init__()
        dropout_value = 0.1
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # 32

        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # 32

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
        ) # 32

        self.P1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16

        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # 16

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # 16

        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # 16

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
        ) # 16

        self.P2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # 8

        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # 8

        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # 6

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # Block 1
        x = self.C1(x)
        x = self.C2(x)
        
        x = self.c1(x)
        x = self.P1(x)
        
        # Block 2
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)

        x = self.c2(x)
        x = self.P2(x)

        # Block 3
        x = self.C6(x)
        x = self.C7(x)
        x = self.C8(x)

        x= self.gap(x)
        x = self.c3(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class NetLight1(nn.Module):
    def __init__(self, normalization) -> None:
        super(NetLight1, self).__init__()
        dropout_value = 0.1
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 16 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 32

        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 32

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0),
        ) # 32

        self.P1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16

        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 16

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 16

        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 16

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0),
        ) # 16

        self.P2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 8

        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 64 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 8

        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            get_normalization(normalization, 64 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 6

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # Block 1
        x = self.C1(x)
        x = self.C2(x)
        
        x = self.c1(x)
        x = self.P1(x)
        
        # Block 2
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)

        x = self.c2(x)
        x = self.P2(x)

        # Block 3
        x = self.C6(x)
        x = self.C7(x)
        x = self.C8(x)

        x= self.gap(x)
        x = self.c3(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class FinalModel(nn.Module):
    def __init__(self, normalization) -> None:
        super(FinalModel, self).__init__()
        dropout_value = 0.1
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 32

        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 32

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0, bias=False),
        ) # 32

        self.P1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16

        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_normalization(normalization, 16 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 16

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_normalization(normalization, 16 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 16

        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_normalization(normalization, 16 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 16

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0, bias=False),
        ) # 16

        self.P2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 8

        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 8

        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_normalization(normalization, 32 , GROUP_SIZE),
            nn.Dropout(dropout_value)
        ) # 6

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        # Block 1
        x = self.C1(x)
        x = self.C2(x)
        
        x = self.c1(x)
        x = self.P1(x)
        
        # Block 2
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)

        x = self.c2(x)
        x = self.P2(x)

        # Block 3
        x = self.C6(x)
        x = self.C7(x)
        x = self.C8(x)

        x= self.gap(x)
        x = self.c3(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
