from torch import nn, unsqueeze

class EEGNet(nn.Module):
    """Model architecture of modified EEGNet"""
    def __init__(
            self, num_class: int, num_channels: int, num_time_points, f_1: int = 8,
            depth: int = 2, f_2: int = 16, drop_out_prob = 0.5, norm = 'layer'
    ):
        """Creation of a randomized model.

        Parameters
        ----------
        num_class : int
            Number of classes
        num_channels : int
            Number of EEG channels
        num_time_points : int
            Number of time points in EEG signal
        f_1 : int, optional
            F_1, by default 8
        depth : int, optional
            Depth, by default 2
        f_2 : int, optional
            F_2, by default 16
        drop_out_prob : float, optional
            Dropout, by default 0.5
        norm : str, optional
            Normalization to be used. Can be either 'batch' or 'layer', by default 'layer'
        """
        super().__init__()
        self.conv2d_1 = nn.Conv2d(
            in_channels=1, out_channels=f_1, kernel_size=(1,num_channels), padding='same'
        )
        """Convolutional Layer"""
        self.norm_1 = nn.LayerNorm([f_1,64,321]) if norm == "layer" else nn.BatchNorm2d(f_1)
        """Normalization Layer"""
        self.depth_wise_conv = nn.Conv2d(
            in_channels=f_1, out_channels=depth*f_1, kernel_size=(num_channels,1),
            padding='valid', groups=f_1
        )
        """Depth wise convolutional Layer"""
        self.norm_2 = nn.LayerNorm([depth*f_1,1,321]) if norm == "layer" else nn.BatchNorm2d(depth*f_1)
        """Normalization Layer"""
        self.elu1 = nn.ELU(alpha=1.0, inplace=True)
        """ELU activation layer"""
        self.avg_pool_2d_1 = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4))
        """Average Pooling layer"""
        self.dropout_1 = nn.Dropout(drop_out_prob)
        """Dropout"""

        self.separable_conv_depth = nn.Conv2d(
            in_channels= depth*f_1, out_channels=depth*f_1, kernel_size=(1,16), padding='same',
            groups=depth*f_1
        )
        """Separable Convolutional layer (depthwise)"""
        self.separable_conv_point = nn.Conv2d(
            in_channels=depth*f_1, out_channels=f_2, kernel_size=(1,1), padding='same'
        )
        """Separable Convolutional layer (pointwise)"""
        self.norm_3 = nn.LayerNorm([f_2,1,80]) if norm == "layer" else nn.BatchNorm2d(f_2)
        """Normalization Layer"""
        self.elu2 = nn.ELU(alpha=1.0, inplace=True)
        """ELU activation layer"""
        self.avg_pool_2d_2 = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8))
        """Average Pooling layer"""
        self.dropout = nn.Dropout(drop_out_prob)
        """Dropout"""
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=f_2*num_time_points//32, out_features=num_class)
    
    def forward(self, x):
        x = unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.norm_1(x)
        x = self.depth_wise_conv(x)
        x = self.norm_2(x)
        x = self.elu1(x)
        x = self.avg_pool_2d_1(x)
        x = self.dropout_1(x)
        x = self.separable_conv_depth(x)
        x = self.separable_conv_point(x)
        x = self.norm_3(x)
        x = self.elu2(x)
        x = self.avg_pool_2d_2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x