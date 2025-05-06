import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from my_models import SingleModel, MyMLP, generate_model
from loader import XclipTrainDataset, XclipTestDataset
from loss_function import topk_loss, center_loss
from eval import evaluate
from efficient_kan import kan


def main(feature_type, model_name, dataset_name):
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cpu')
    epochs = 10

    if feature_type == 'feature_v' and model_name == 'SingleModel':
        model = generate_model('SingleModel', 512).to(device)
    elif feature_type == 'feature_v' and model_name == 'MyKan':
        model = generate_model('MyKan', 512).to(device)
    elif feature_type == 'y' and model_name == 'MyMLP':
        model = generate_model('MyMLP', 2).to(device)
    elif feature_type == 'y' and model_name == 'MyKan':
        model = generate_model('MyKan', 2).to(device)
    else:
        raise ValueError('model name is out of options!!!')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    if dataset_name == 'TAD':
        train_file_path = './features/TAD/train_features_xclip.pkl'
        train_labels_path = './features/TAD/video_level_labels.pkl'
        train_data = XclipTrainDataset(400, train_file_path, train_labels_path)
        test_file_path = './features/TAD/test_features_xclip.pkl'
        test_labels_path = './labels/TAD_test.txt'
        test_data = XclipTestDataset(100, test_file_path, test_labels_path)
        pretrained = './features/TAD/pretrained_model.pth'

    elif dataset_name == 'SHT':
        train_file_path = './features/SHT/train_features_xclip.pkl'
        train_labels_path = './features/SHT/video_level_labels.pkl'
        train_data = XclipTrainDataset(238, train_file_path, train_labels_path)  # 238个训练视频
        test_file_path = './features/SHT/test_features_xclip.pkl'
        test_labels_path = './labels/Shanghaitech_test.txt'
        test_data = XclipTestDataset(199, test_file_path, test_labels_path)  # 199个测试视频
        pretrained = './features/SHT/pretrained_model.pth'

    else:
        raise ValueError('name of dataset is out of options!!!')

    train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)

    if not os.path.exists(pretrained):
        # train
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, labels) in enumerate(train_loader):
                features = data[feature_type]
                instances_logits = model(features)
                total_loss = topk_loss(instances_logits, labels, 3, device) + 10 * center_loss(instances_logits, labels, device)
                epoch_loss += total_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            loss_avg = epoch_loss / (batch_idx + 1)
            print(f'epoch: {epoch}; loss: {loss_avg}')
        torch.save(obj=model, f=pretrained)
        print('training finished!!!')

    else:
        # test
        model = torch.load(pretrained)
        model.eval()

        eval_dict = dict()
        for batch_idx, (data, v_name) in enumerate(test_loader):
            features = data[feature_type]
            instances_logits = model(features)
            instances_logits_np = instances_logits.detach().cpu().numpy()
            instances_logits_np = instances_logits_np.squeeze(0).squeeze(1)
            # [1, 9, 1] => [9, 1] => [9,] axis=1
            try:
                if isinstance(v_name, tuple):
                    v_name = v_name[0]
            except ValueError as e:
                raise Exception('the data type is not right!', e)

            eval_dict[v_name] = [instances_logits_np]

        result = evaluate(eval_dict, test_labels_path, root='')
        print(f' the total AUC is: {result[0]},\n AUC of anomalies is: {result[1]}')


if __name__ == '__main__':
    # main(feature_type='y', model_name='MyKan', dataset_name='SHT')
    # main(feature_type='feature_v', model_name='MyKan', dataset_name='SHT')
    main(feature_type='y', model_name='MyKan', dataset_name='TAD')







