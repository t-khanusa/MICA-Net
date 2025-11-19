from __future__ import print_function
from __future__ import division
import numpy as np
from torch.utils.data import Dataset
import time
import copy
from model.MICA_Net import MICA_Net


def normalize(modality):
    x = np.zeros((1, len(modality[0])))
    max = float(np.max(modality[0]))
    min = float(np.min(modality[0]))
    for i in range(len(modality[0])):
        x[0][i] = float((float(modality[0][i]) - min) / (max - min))

    return x


# Creating custom Dataset classes
class MyDataset(Dataset):
    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = target

    def __getitem__(self, index):
        # labels = self.target)
        z = self.target[index]
        z = np.array(z, int)
        z = torch.from_numpy(z)

        samples = self.data_path
        x = np.load(samples[index][0])[0]
        x = (x.transpose(1, 2, 0))[0]
        # x = normalize(x, 0.2)
        x = torch.from_numpy(x).float()
        # x = torch.flatten(x)

        # y = np.load(samples[index][1])[0][0][0]
        y = np.load(samples[index][1])
        # y = normalize(y, 0.8)
        y = torch.from_numpy(y).float()
        # y = torch.flatten(y)
        return x, y, z

    def __len__(self):
        return len(self.data_path)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    print("Start training !")
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for input1, input2, labels in dataloaders[phase]:
                input1 = input1.to(device)
                input2 = input2.to(device)
                labels = labels.to(device)
                # print(input1.size(0))
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(input1, input2)
                    # print("outputs: ", outputs.shape)
                    # print("labels ", labels.shape)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # print("inputs size: ", inputs.size(0))
                # print("loss item: ", loss.item())
                running_loss += loss.item() * input1.size(0)
                # print("running loss: ", running_loss)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    import glob, os
    import torch

    failed_path = []

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    Inertial_data = "/content/drive/MyDrive/Q1_paper/UESTC_feature/video_pytorch"
    Video_data = "/content/drive/MyDrive/Q1_paper/UESTC_feature/inertial"

    for inertial_path in glob.glob(os.path.join(Inertial_data, 'train', '*', '*.npy')):
        if inertial_path[-8:] == '.mp4.npy':
            pass
        else:
            label = int(inertial_path.split("/")[-2]) - 1
            video_path = inertial_path.replace("video_pytorch", "inertial")
            train_x.append([video_path, inertial_path])
            train_y.append(label)

    for inertial_path in glob.glob(os.path.join(Inertial_data, 'val', '*', '*.npy')):
        if inertial_path[-8:] == '.mp4.npy':
            pass
        else:
            label = int(inertial_path.split("/")[-2]) - 1
            video_path = inertial_path.replace("video_pytorch", "inertial")
            val_x.append([video_path, inertial_path])
            val_y.append(label)

    for inertial_path in glob.glob(os.path.join(Inertial_data, 'test', '*', '*.npy')):
        if inertial_path[-8:] == '.mp4.npy':
            pass
        else:
            label = int(inertial_path.split("/")[-2]) - 1
            video_path = inertial_path.replace("video_pytorch", "inertial")
            test_x.append([video_path, inertial_path])
            test_y.append(label)

    batch_size = 16
    print("train: ", len(train_y))
    print("val: ", len(val_y))
    print("test: ", len(test_y))

    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(MyDataset(train_x, train_y), batch_size=batch_size, shuffle=True,
                                             num_workers=4),
        'val': torch.utils.data.DataLoader(MyDataset(val_x, val_y), batch_size=batch_size, shuffle=False,
                                           num_workers=4),
        'test': torch.utils.data.DataLoader(MyDataset(test_x, test_y), batch_size=batch_size, shuffle=False,
                                            num_workers=4)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cam = MICA_Net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cam.parameters(), 0.0001)
    # optimizer = torch.optim.SGD(cam.parameters(), lr=0.0001, momentum=0.9)
    _, scratch_hist = train_model(cam, dataloaders_dict, criterion, optimizer, num_epochs=15)
