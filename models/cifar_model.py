import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarModel(nn.Module):
    """"
    The ANN build using PyTorch library.
    """

    def __init__(self):
        """
        The class constructor.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3))
        self.batch_norm = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.25)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        The forward pass.
        :return: None
        """
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(x)
        x = x.view(x.size(0), -1)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.softmax(x)

        return x

    def training_step(self, batch):
        """
        The training step
        :param batch:
        :return:
        """
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    @staticmethod
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = self.accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    @staticmethod
    def validation_epoch_end(outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    @staticmethod
    def epoch_end(epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    def evaluate(self, val_loader):
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)

    def fit(self, epochs, lr, train_loader, val_loader):
        history = []
        optimizer = torch.optim.Adam(self.parameters(), lr, amsgrad=True)
        for epoch in range(epochs):
            # Training Phase
            for batch in train_loader:
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = self.evaluate(self, val_loader)
            self.epoch_end(epoch, result)
            history.append(result)
        return history

