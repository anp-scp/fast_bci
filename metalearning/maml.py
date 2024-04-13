from torch import no_grad, Tensor, autograd, eq, sigmoid, unsqueeze, save, load, optim
from torch.nn import functional as F, Module, ParameterList
from torch.utils.data import Dataset, DataLoader
from metalearning import constants
from tqdm.auto import tqdm
from copy import deepcopy
from typing import Union
from dvclive import Live
import numpy as np
import os

class MAMLModel(Module):
    """Class of `torch.nn.Module` objects that are to be trained using MAML.
    The architecture needs to be defined as list of tuples. And other
    attributes need to be defines as shown in the following example:
    
    ```python
    class ForMiniimagenet(MAMLModel):
        def __init__(self):
            super().__init__()
            self.config = [
                ('conv2d', [32, 3, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 1, 0]),
                ('flatten', []),
                ('linear', [5, 32 * 5 * 5])
            ]
            self.vars, self.vars_bn = modelParametersInitializer(architecture=self.config)
    ```
    """

    def __init__(self):
        super().__init__()
        # this dict contains all tensors needed to be optimized
        self.vars: Union[ParameterList,list] = ParameterList()
        """parameters of model"""
        # running_mean and running_var
        self.vars_bn: Union[ParameterList,list] = ParameterList()
        """running_mean and running_var for batch norm"""
    
    def forward(self, x: Tensor, vars: Union[ParameterList, list], bn_training: bool = True) -> Tensor:
        """Method to run forward pass

        Parameters
        ----------
        x : Tensor
            input
        vars : Union[ParameterList, list]
            parameters (or fast parameters)
        bn_training : bool, optional
            To enable of disable batch norm training, by default True

        Returns
        -------
        Tensor
            Output

        Raises
        ------
        NotImplementedError
            Raised when architecture component is not defined
        """              
        if vars is None:
            vars = self.vars
        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5], groups=param[6])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5], groups=param[6])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'ln':
                w, b = vars[idx], vars[idx + 1]
                x = F.layer_norm(x, tuple(w.shape), w, b)
                idx += 2

            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name == 'dropout':
                x = F.dropout(x, param[0], bn_training)
            elif name == 'unsqueeze':
                x = unsqueeze(x, param[0])
            elif name == 'elu':
                x = F.elu(x, param[0], param[1])
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        return x
    
    def extra_repr(self) -> str:
        """Method for custom representation

        Returns
        -------
        str
            string representation

        Raises
        ------
        NotImplementedError
            Raised when architecture component is not defined
        """        
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%s, groups:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], str(param[5]), param[6])
                info += tmp + '\n'
            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%s, groups:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], str(param[5]), param[6])
                info += tmp + '\n'
            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'
            elif name == 'avg_pool2d':
                tmp = f'avg_pool2d:(k:{param[0]}, stride:{param[1]}, padding:{param[2]})'
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = f'max_pool2d:(k:{param[0]}, stride:{param[1]}, padding:{param[2]})'
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 'ln', 'dropout', 'unsqueeze', 'elu']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info
    
    def zero_grad(self, vars: Union[ParameterList, list] = None):
        """Custom method to make gradients of custom parameters zero

        Parameters
        ----------
        vars : Union[ParameterList, list], optional
            Parameters (or fast parameters), by default None
        """              
        with no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self) -> Union[ParameterList, list]:
        """override this function since initial parameters will return with a generator.

        Returns
        -------
        Union[ParameterList, list]
            The parameters of the model
        """        
        return self.vars

class MAMLHandler:
    """Wrapper class to train a model using Meta Agnostic Meta Learning
    """    
    def __init__(
            self, model: Module, device: int, trainDataLoader: DataLoader,
            validationDataLoader: DataLoader, updateStepsInInnerLoopTrain: int,
            updateStepsInInnerLoopValid: int, adaptLearningRate: float, epochs: int,
            optimizer, live: Live, modelSaveDir: str = None, modelName: str = None
    ):
        """
        Creating an object of MAMLHandler class.

        Parameters
        ----------
        model : Module
            Model to be trained
        device : int
            device id
        trainDataLoader : DataLoader
            train data loader
        validationDataLoader : DataLoader
            validation data loader
        updateStepsInInnerLoopTrain : int
            number of gradient update steps while training
        updateStepsInInnerLoopTest : int
            number of gradient update steps while validation
        adaptLearningRate : float
            learning while adapting (inner loop)
        epochs : int
            number of epochs
        optimizer : torch.optim.Optimizer
            Optimizer to be used
        live : dvclive.Live
            A DVC live object for logging
        modelSaveDir : str, optional
            dir path to stoe model, by default None
        modelName : str, optional
            Name of model when saved
        """             
        self.model: Module = model
        """Provided model during object creation"""
        self.device: int = device
        """Provided GPU device"""
        self.trainLoader: DataLoader = trainDataLoader
        """Provided train dataloader"""
        self.validationLoader: DataLoader = validationDataLoader
        """Provided validation dataloader"""
        self.stepsInInnerLoopTrain: int = updateStepsInInnerLoopTrain
        """Number of update steps provided for inner loop while training"""
        self.stepsInInnerLoopValid: int = updateStepsInInnerLoopValid
        """Number of update steps provided for inner loop while validation"""
        self.adaptLearningRate: float = adaptLearningRate
        """Provided learning rate for inner loop"""
        self.epochs: int = epochs
        """Provided number of epochs"""
        self.meta_optim: optim.Optimizer = optimizer
        """Provided torch optimizer"""
        self.modelName: str = modelName
        """Provided name of the model file"""
        self.modelSaveDir: str = modelSaveDir
        """Provided path to directory to sabe the model"""
        self.live: Live = live
        """Provided DVC live object"""
    
    def __performAction(
            self, actionType:str, X_support: Tensor, y_support: Tensor, X_query: Tensor,
            y_query: Tensor, adapt_steps: int, get_fine_tune_metrics: bool = False
    ) -> list:

        assert actionType in [constants.VAL, constants.FINETUNE, constants.TRAIN]
        task_num = X_support.size(0)
        supportsz = X_support.size(1)
        querysz = X_query.size(1)

        X_support, y_support, X_query, y_query = X_support.to(self.device),\
            y_support.to(self.device), X_query.to(self.device), y_query.to(self.device)

        model = self.model
        batch_norm_query = True
        if actionType in [constants.VAL, constants.FINETUNE]:
            model = deepcopy(model)
            batch_norm_query = False
        model = model.to(self.device)

        #losses_q[i] is the loss on step i
        losses_q = [0 for _ in range(adapt_steps + 1)] 
        corrects = [0 for _ in range(adapt_steps + 1)]
        if get_fine_tune_metrics:
            corrects_support = [0 for _ in range(adapt_steps + 1)]    

        for i in range(task_num):
            logits = model(X_support[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_support[i])
            grad = autograd.grad(loss, model.parameters())
            fast_weights = list(map(
                lambda p: p[1] - self.adaptLearningRate * p[0], zip(grad, model.parameters())
            ))

            # this is the loss and accuracy before first update
            with no_grad():
                # [setsz, nway]
                if get_fine_tune_metrics:
                    pred_support = F.softmax(logits, dim=1).argmax(dim=1)
                    corrects_support[0] = corrects_support[0] + eq(pred_support, y_support[i]).sum().item()
                logits_q = model(X_query[i], model.parameters(), bn_training=batch_norm_query)
                loss_q = F.cross_entropy(logits_q, y_query[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = eq(pred_q, y_query[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with no_grad():
                # [setsz, nway]
                if get_fine_tune_metrics:
                    logits = model(X_support[i], vars=fast_weights, bn_training=False)
                    pred_support = F.softmax(logits, dim=1).argmax(dim=1)
                    corrects_support[1] = corrects_support[1] + eq(pred_support, y_support[i]).sum().item()
                logits_q = model(X_query[i], fast_weights, bn_training=batch_norm_query)
                loss_q = F.cross_entropy(logits_q, y_query[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = eq(pred_q, y_query[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, adapt_steps):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = model(X_support[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_support[i])
                # 2. compute grad on theta_pi
                grad = autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(
                    lambda p: p[1] - self.adaptLearningRate * p[0], zip(grad, fast_weights)
                ))

                logits_q = model(X_query[i], fast_weights, bn_training=batch_norm_query)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_query[i])
                losses_q[k + 1] += loss_q

                with no_grad():
                    if get_fine_tune_metrics:
                        pred_support = F.softmax(logits, dim=1).argmax(dim=1)
                        corrects_support[k+1] = corrects_support[k+1] + eq(pred_support, y_support[i]).sum().item()
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = eq(pred_q, y_query[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        if actionType == 'train':
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        if actionType == constants.VAL:
            del model
            if get_fine_tune_metrics:
                accs_support = np.array(corrects_support) / (supportsz * task_num)
                return accs, accs_support
            return accs
        if actionType == constants.FINETUNE:
            return  model,accs
        return accs
    
    def train(self):
        """Method to start the MAML training
        """        
        maxValidationAccSoFar = np.NINF
        for epoch in tqdm(range(self.epochs), desc='Epochs'):
            acc_all_train = [] 
            for metaStep, (X_support, y_support, X_query, y_query) in enumerate(self.trainLoader):
                X_support = X_support.to(self.device)
                y_support = y_support.to(self.device)
                X_query = X_query.to(self.device)
                y_query = y_query.to(self.device)

                ## performe a meta update
                trainAcc = self.__performAction(
                    constants.TRAIN, X_support, y_support, X_query, y_query, self.stepsInInnerLoopTrain
                )
                acc_all_train.append(trainAcc)
                if metaStep % 5 == 0:
                    print('step:', metaStep, '\ttraining acc:', trainAcc)

                if metaStep % 5 == 0:  # evaluation or validation
                    accs_all_validation = []
                    for x_spt, y_spt, x_qry, y_qry in self.validationLoader:
                        accs = self.__performAction(constants.VAL, x_spt, y_spt, x_qry, y_qry, self.stepsInInnerLoopValid)
                        accs_all_validation.append(accs)

                    accs = np.array(accs_all_validation).mean(axis=0).astype(np.float16)
                    print('Validation acc:', accs)
            accs_all_validation = []
            for x_spt, y_spt, x_qry, y_qry in self.validationLoader:
                accs = self.__performAction(constants.VAL, x_spt, y_spt, x_qry, y_qry, self.stepsInInnerLoopValid)
                accs_all_validation.append(accs)
            acc = np.array(acc_all_train).mean(axis=0).astype(np.float16)[-1]
            self.live.log_metric("training_acc_vs_epoch", acc)
            acc = np.array(accs_all_validation).mean(axis=0).astype(np.float16)[-1]
            self.live.log_metric("validation_acc_vs_epoch", acc)
            if acc > maxValidationAccSoFar:
                maxValidationAccSoFar = acc
                save(self.model.state_dict(), os.path.join(self.modelSaveDir, self.modelName + ".pt"))
            print(f"Current Validation Accuracy: {acc}")
            print(f"Max Validation Accuracy so far: {maxValidationAccSoFar}")
            self.live.next_step()
        self.model.load_state_dict(load(os.path.join(self.modelSaveDir, self.modelName + ".pt"), map_location=f"{self.device}"))
        self.live.log_metric("best_validation_acc", maxValidationAccSoFar, plot=False)
    
    def test(self, test_data: DataLoader, adapt_steps: int = None):
        """Check performance on test data

        Parameters
        ----------
        test_data : DataLoader
            The test data
        adapt_steps : int, optional
            Number of steps for adaptation. If None then same as that in validation, by default None

        Returns
        -------
        tuple
            Tuple containing test accuracies and training accuracies during finetuning at  each
            iteration. Both the elements of the tuple is a list. Also the accuracies are avreaged
            over all the provided subjects.
        """
        self.model.load_state_dict(load(os.path.join(self.modelSaveDir, self.modelName + ".pt"), map_location=f"{self.device}"))
        adapt_steps = self.stepsInInnerLoopValid if adapt_steps is None else adapt_steps
        acc_all_finetune = []
        acc_all_test = []
        for metaStep, (X_support, y_support, X_query, y_query) in enumerate(test_data):
                X_support = X_support.to(self.device)
                y_support = y_support.to(self.device)
                X_query = X_query.to(self.device)
                y_query = y_query.to(self.device)

                acc = self.__performAction(
                    constants.VAL, X_support, y_support, X_query, y_query, adapt_steps, True
                )
                acc_all_test.append(acc[0])
                acc_all_finetune.append(acc[1])
        return np.array(acc_all_test).mean(axis=0).astype(np.float16), np.array(acc_all_finetune).mean(axis=0).astype(np.float16)