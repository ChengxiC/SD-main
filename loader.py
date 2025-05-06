import torch.utils.data as data
import pickle
import os


def read_pickle_file(file_path):

    if file_path == '':
        raise ValueError("it is not a path!!!")
    elif not os.path.isfile(file_path):
        raise ValueError("not a file!!!")
    try:
        with open(file_path, 'rb') as file:
            dict_file = pickle.load(file)
            return dict_file
    except pickle.PickleError:
        raise pickle.PickleError(f"the path of {file_path} not found!!!")


class XclipTrainDataset(data.Dataset):

    def __init__(self, num_sample, train_features_path, train_labels_path):
        self.num_sample = num_sample
        self.train_features_path = train_features_path
        self.train_labels_path = train_labels_path

    def __getitem__(self, item):
        assert item < self.num_sample
        dict_features = read_pickle_file(self.train_features_path)
        bag_labels = read_pickle_file(self.train_labels_path)
        return dict_features[item], bag_labels[item]

    def __len__(self):
        return self.num_sample


class XclipTestDataset(data.Dataset):

    def __init__(self, num_sample, test_features_path, test_labels_path):
        self.num_sample = num_sample
        self.test_features_path = test_features_path
        self.test_labels_path = test_labels_path

    def get_id_name(self):

        vid_dict = {}
        video_id = 0
        with open(self.test_labels_path, 'r') as fin:
            for line in fin:
                linespit = line.strip().split()
                vid_dict[video_id] = linespit[0]
                video_id += 1
        return vid_dict

    def __getitem__(self, item):
        assert item < self.num_sample
        vid_dict = self.get_id_name()
        if item in vid_dict.keys():
            vid_name = vid_dict[item]
        else:
            raise ValueError('the key is wrong!!!')

        dict_features = read_pickle_file(self.test_features_path)

        return dict_features[item], vid_name

    def __len__(self):
        return self.num_sample
