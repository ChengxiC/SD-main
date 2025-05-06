import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from efficient_kan import kan


def weights_init(m):

    class_name = m.__class__.__name__

    if class_name.find('Conv') != -1 or class_name.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class SingleModel(torch.nn.Module):

    def __init__(self, n_feature):
        super(SingleModel, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, 1)
        self.dropout = nn.Dropout(0.0)
        self.apply(weights_init)

    def forward(self, inputs, train=True):
        x = F.relu(self.fc(inputs))
        if train:
            x = self.dropout(x)
        return F.sigmoid(self.classifier(x))


class MyMLP(nn.Module):

    def __init__(self, num_features):
        super(MyMLP, self).__init__()
        self.fc = nn.Linear(num_features, 128)
        self.classifier = nn.Linear(128, 1)
        self.apply(weights_init)

    def forward(self, x, train=True):
        x = F.relu(self.fc(x))
        x = self.classifier(x)
        return F.sigmoid(x)


def generate_model(model_name, num_features):
    if model_name == 'MyMLP':
        model = MyMLP(num_features)
        return model
    elif model_name == 'SingleModel':
        model = SingleModel(num_features)
        return model
    elif model_name == 'MyKan':
        model = kan.KAN([num_features, 64, 1])
        return model
    else:
        raise RuntimeError('out of options!')





