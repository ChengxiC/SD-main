import torch
from einops import rearrange
import pickle
import numpy as np
import os


def test_for_traindata(train_data_path=''):

    try:
        with open(train_data_path, 'rb') as file:
            load_dict = pickle.load(file)

        query = input('please input an int number:')
        for k, v in load_dict.items():
            if isinstance(v, dict):
                if k == str(query):
                    for keys, v_tensor in v.items():
                        print(keys, v_tensor)
                        print('yes') if isinstance(keys, str) else print('no')
            else:
                raise ValueError('the feature dictionary is something wrong!!!')

    except (pickle.PickleError, FileNotFoundError) as e:
        raise Exception('check the path!!!', e)


def test_for_testdata(test_data_path=''):

    try:
        with open(test_data_path, 'rb') as file:
            load_dict = pickle.load(file)

        print([k for k in load_dict.keys()])
        print('the number of the keys is： ', len(load_dict))
        print(f'the number of clips in videos is: {[(len(v), type(v)) for v in load_dict.values()]}')

    except (pickle.PickleError, FileNotFoundError) as e:
        raise Exception('check the path!!!', e)


@torch.no_grad()
def get_features(model, data_loader, text_labels, device, saved_path, mode, config):

    model.eval()
    texts = text_labels.to(device)
    features_xclip = dict()

    if mode == 'train':

        for idx, batch_data in enumerate(data_loader):
            images = batch_data["imgs"].to(device)[:, :1]
            images = rearrange(images, 'b a k c t h w -> (b a k) t c h w')  # bz*num_aug*num_clips, num_frames, c, h, w
            if texts.shape[0] == 1:
                texts = texts.view(1, -1)

            output = model(images, texts)

            # for one batch, the train output:  output['y'] [32, 2]; output['y_cluster_all'] [32, 2];
            # output['feature_v'] [32, 512]; output['y_cluster_all_nograd'] [32, 2]
            output1_cpu = {k: v[:16, :].detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
            output2_cpu = {k: v[16:, :].detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
            output1_cpu = {int(batch_data["vid"][0].item()): output1_cpu}
            output2_cpu = {int(batch_data["vid"][1].item()): output2_cpu}
            features_xclip.update(output1_cpu)
            features_xclip.update(output2_cpu)
            print(f'batch: {idx}, dictionary updated!')

    elif mode == 'test':

        video_list = []

        with open(config.DATA.VAL_FILE, 'r') as fin:
            for line in fin:
                linespit = line.strip().split()
                video_list.append(linespit[0])

        list_feature_v = []
        list_y = []
        first_id = 0
        num_batches = len(data_loader)
        for idx, batch_data in enumerate(data_loader):

            images = batch_data["imgs"].to(device)
            b, k, c, t, h, w = images.size()
            images = rearrange(images, 'b k c t h w -> (b k) t c h w')
            output = model(images, texts)  # batch = 64

            feature_v_np = output['feature_v'].cpu().data.numpy()
            feature_y_np = output['y'].cpu().data.numpy()

            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
            #  2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5,
            #  5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6]
            if idx < num_batches - 1:
                for ind in range(b):
                    if first_id == batch_data["vid"][ind].item():
                        list_feature_v.append(feature_v_np[ind])
                        list_y.append(feature_y_np[ind])
                        if first_id not in features_xclip.keys():
                            features_xclip[first_id] = {}
                    elif first_id != batch_data["vid"][ind].item():
                        features_xclip[first_id]['feature_v'] = np.stack(list_feature_v)
                        features_xclip[first_id]['y'] = np.stack(list_y)
                        list_y = []
                        list_feature_v = []
                        first_id += 1
            elif idx == num_batches - 1:
                for ind in range(b):
                    if first_id == batch_data["vid"][ind].item() and ind < b-1:
                        list_feature_v.append(feature_v_np[ind])
                        list_y.append(feature_y_np[ind])
                        if first_id not in features_xclip.keys():
                            features_xclip[first_id] = {}
                    elif first_id != batch_data["vid"][ind].item() and ind < b-1:
                        features_xclip[first_id]['feature_v'] = np.stack(list_feature_v)
                        features_xclip[first_id]['y'] = np.stack(list_y)
                        list_y = []
                        list_feature_v = []
                        first_id += 1
                    elif first_id == batch_data["vid"][ind].item() and ind == b-1:
                        list_feature_v.append(feature_v_np[ind])
                        list_y.append(feature_y_np[ind])
                        features_xclip[first_id]['feature_v'] = np.stack(list_feature_v)
                        features_xclip[first_id]['y'] = np.stack(list_y)
                    else:
                        pass

            print(f'batch: {idx}, finished!')

    else:
        raise ValueError('out of options!!!!!')

    try:
        with open(saved_path, 'wb') as file:
            pickle.dump(features_xclip, file)
    except FileNotFoundError:
        raise FileNotFoundError('check the path file!!!')

    print('get all features!')


def get_video_labels(data_loader, device, saved_path):

    video_level_labels = {}
    for idx, batch_data in enumerate(data_loader):

        label_id = batch_data["label"].to(device)[:, :1]
        label_id = label_id.reshape(-1)

        v1 = 1 if label_id[0].item() >= 1 else 0
        v2 = 1 if label_id[1].item() >= 1 else 0
        label1_dict = {int(batch_data["vid"][0].item()): v1}
        label2_dict = {int(batch_data["vid"][1].item()): v2}
        video_level_labels.update(label1_dict)
        video_level_labels.update(label2_dict)

        print(f'batch: {idx}, dictionary updated!')

    try:
        with open(saved_path, 'wb') as file:
            pickle.dump(video_level_labels, file)
    except:
        raise FileNotFoundError('check the path file!!!')

    print('get all video labels!')


def get_frame_video_labels(eval_path, saved_path_f, saved_path_v):

    frame_labels = {}
    video_labels = []
    vid = 0
    if eval_path is None or not os.path.isfile(eval_path):
        raise FileNotFoundError('check the test path!!!')
    with open(eval_path, 'r') as file:
        for line in file:
            line_split = line.strip().split()
            if len(line_split) != 5:
                raise ValueError('this is not ground truth frame labels!!!')
            filename, end_ind, label, anomaly_start_ind, anomaly_end_ind = line_split
            try:
                end_ind = int(end_ind)
                label = int(label)
                anomaly_start_ind = int(anomaly_start_ind)
                anomaly_end_ind = int(anomaly_end_ind)
                # 确保各个值在合理范围
                assert 0 <= anomaly_start_ind <= end_ind
                assert 0 <= anomaly_end_ind <= end_ind
                assert anomaly_start_ind < anomaly_end_ind
            except (ValueError, AssertionError) as e:
                raise Exception(f'something wrong with your file {e}')

            if label == 0:
                frame_label_lst = np.zeros(end_ind)
                frame_labels[vid] = frame_label_lst
                video_labels.append(0)
            elif label > 0:
                frame_label_lst = np.zeros(end_ind)
                frame_label_lst[anomaly_start_ind:anomaly_end_ind] = 1
                frame_labels[vid] = frame_label_lst
                video_labels.append(1)
            else:
                raise ValueError('the label is not right!!!')
            vid += 1

    try:
        with open(saved_path_f, 'wb') as file1:
            pickle.dump(obj=frame_labels, file=file1)

        with open(saved_path_v, 'wb') as file2:
            pickle.dump(obj=video_labels, file=file2)
    except FileNotFoundError:
        raise Exception('check the file path!!!')

