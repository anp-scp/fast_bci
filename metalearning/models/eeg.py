from metalearning.maml import MAMLModel
from metalearning.utils import modelParametersInitializer

class EEGNet(MAMLModel):
    """Model architecture for modified EEGNet to be trained using MAML
    """
    def __init__(
            self, num_class: int, num_channels: int, num_time_points: int, f_1: int = 8,
            depth: int = 2, f_2: int = 16, drop_out_prob = 0.5
    ):
        """
        Creation of an randomized EEGNet model

        Parameters
        ----------
        num_class : int
            Number of classes/labels
        num_channels : int
            Number of EEG channels
        num_time_points : int
            Number of time points in EEG
        f_1 : int, optional
            F_1, by default 8
        depth : int, optional
            Depth, by default 2
        f_2 : int, optional
            F_2, by default 16
        drop_out_prob : float, optional
            dropout, by default 0.5
        """
        super().__init__()
        self.config = [
            ('unsqueeze', [1]),
            # c_out = f_1, c_in = 1, kernel_size = (1,80), strides=1, padding='same', groups = 1
            ('conv2d', [f_1, 1, 1, 64, 1, 'same', 1]),
            ('ln', [f_1,64,321]),
            # depth wise convolution
            # c_out = depth*f_1, c_in = f_1, kernel_size = (C,1), strides=1, padding='valid',
            # groups = 1, groups = c_in
            ('conv2d', [depth*f_1, f_1, num_channels, 1, 1, 'valid', f_1]),
            ('ln', [depth*f_1,1,321]),
            # alpha = 1.0, inplace = True
            ('elu', [1.0, True]),
            ('avg_pool2d', [(1,4), (1,4), 0]),
            ('dropout', [drop_out_prob]),
            #separable convolution part1 depthwise
            ('conv2d', [depth*f_1, depth*f_1, 1, 16, 1, 'same', depth*f_1]),
            #separable convolution part1 pointwise
            ('conv2d', [f_2, depth*f_1, 1, 1, 1, 'same', 1]),
            ('ln', [f_2,1,80]),
            ('elu', [1.0, True]),
            ('avg_pool2d', [(1,8), (1,8), 0]),
            ('dropout', [drop_out_prob]),
            ('flatten', []),
            ('linear', [num_class, f_2*num_time_points//32])
        ]
        """Configuration of model architecture for MAML:
        ```python
        [
            ('unsqueeze', [1]),
            # c_out = f_1, c_in = 1, kernel_size = (1,80), strides=1, padding='same', groups = 1
            ('conv2d', [f_1, 1, 1, 64, 1, 'same', 1]),
            ('ln', [f_1,64,321]),
            # depth wise convolution
            # c_out = depth*f_1, c_in = f_1, kernel_size = (C,1), strides=1, padding='valid',
            # groups = 1, groups = c_in
            ('conv2d', [depth*f_1, f_1, num_channels, 1, 1, 'valid', f_1]),
            ('ln', [depth*f_1,1,321]),
            # alpha = 1.0, inplace = True
            ('elu', [1.0, True]),
            ('avg_pool2d', [(1,4), (1,4), 0]),
            ('dropout', [drop_out_prob]),
            #separable convolution part1 depthwise
            ('conv2d', [depth*f_1, depth*f_1, 1, 16, 1, 'same', depth*f_1]),
            #separable convolution part1 pointwise
            ('conv2d', [f_2, depth*f_1, 1, 1, 1, 'same', 1]),
            ('ln', [f_2,1,80]),
            ('elu', [1.0, True]),
            ('avg_pool2d', [(1,8), (1,8), 0]),
            ('dropout', [drop_out_prob]),
            ('flatten', []),
            ('linear', [num_class, f_2*num_time_points//32])
        ]
        ```
        """
        self.vars, self.vars_bn = modelParametersInitializer(architecture=self.config)
