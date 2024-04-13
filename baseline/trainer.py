from tqdm import tqdm
from dvclive import Live
from typing import Callable
from torch.optim import Optimizer
from torch import nn, save, no_grad, eq
from torch.utils.data import DataLoader


class TrainerSingleGPU:
    """Wrapper for training DNN model"""
    def __init__(
        self, model: nn.Module, train_data: DataLoader, valid_data: DataLoader,
        optimizer: Optimizer, gpu_id: int,
        loss_criterion: Callable, snap_shot_path: str, live: Live
    ) -> None:
        """Creation of an object of TrainerSingleGPU class

        Parameters
        ----------
        model : nn.Module
            Pytorch Model
        train_data : DataLoader
            Train Dataloader
        valid_data : DataLoader
            Validation Dataloader
        optimizer : Optimizer
            Optimizer for updating parameters
        gpu_id : int
            GPU device ID
        loss_criterion : Callable
            The loss function
        snap_shot_path : str
            Path to a file where the model needs to be stored
        live : Live
            A DVC Live object for logging
        """
        self.gpu_id: int = gpu_id
        """Provided GPU device id"""
        self.model: nn.Module = model
        """Provided Pytorch model"""
        self.train_data: DataLoader = train_data
        """Provided train Dataloader"""
        self.valid_data: DataLoader = valid_data
        """Provided validation Dataloader"""
        self.optimizer: Optimizer = optimizer
        """Provided optimixer to update parameters"""
        self.epochs_run: int = 0
        """Stores the currently running epoch"""
        self.loss_criterion: Callable = loss_criterion
        """Provided loss function"""
        self.best_snapshot_path: str = snap_shot_path
        """The path to store the best snap shot path"""
        self.best_acc: float = float('-inf')
        """Stores the best validation accuracy"""
        self.live: Live = live
        """Provided DVCLive object for logging"""
        self.model = self.model.to(gpu_id)

    def _run_epoch(self, epoch):
        self.model.train()
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Training at Epoch {epoch} | Batchsize: {b_sz}")
        total, correct = 0,0
        for _, (source, targets) in enumerate(tqdm(self.train_data, desc=f"Training in batches at epoch {epoch}")):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self.optimizer.zero_grad()
            output = self.model(source)
            loss = self.loss_criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            y_hat = nn.functional.softmax(output,1).argmax(1)
            correct += eq(targets, y_hat).sum().item()
            total += len(targets)
        epoch_train_acc = correct / total
        self.live.log_metric(
            'training_acc_vs_epoch', epoch_train_acc
        )
        if epoch % 30 == 0:
            print(f"Training accuracy at epoch {epoch} is {epoch_train_acc}")

    
    def _run_validation(self, epoch):
        self.model.eval()
        print(f"[GPU{self.gpu_id}] Validating at Epoch {epoch}")
        total, correct = 0,0
        with no_grad():
            for _, (source, targets) in enumerate(tqdm(self.valid_data, desc=f"Validating at epoch {epoch}")):
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)
                y_hat = nn.functional.softmax(output, 1).argmax(1)
                correct += eq(targets, y_hat).sum().item()
                total += len(targets)
            epoch_valid_acc = correct / total
            self.live.log_metric(
                'validation_acc_vs_epoch', epoch_valid_acc
            )
            if epoch_valid_acc > self.best_acc:
                self.best_acc = epoch_valid_acc
                self._save_snapshot(epoch, self.best_snapshot_path)


    def _save_snapshot(self, epoch, path):
        snapshot = {
            "BEST_ACC": self.best_acc,
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        save(snapshot, path)
        print(f"Epoch {epoch} | Training snapshot saved at {path}")

    def train(self, max_epochs: int):
        """Train the model.

        Parameters
        ----------
        max_epochs : int
            Maximum epoch

        Returns
        -------
        float
            Best validation accuracy
        """
        for epoch in tqdm(range(self.epochs_run, max_epochs), desc="EPOCHS"):
            self._run_epoch(epoch)
            self._run_validation(epoch)
            self.live.next_step()
        return self.best_acc
