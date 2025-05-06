import os
import pickle
import torch
import torch.backends.cudnn as cudnn
import argparse
from pathlib import Path
from utils.tools import generate_text
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from utils.config import get_config
from models import xclip
from get_method import get_features, get_frame_video_labels, get_video_labels


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opts",  default=None, nargs='+',  help="Modify config options by adding 'KEY VALUE' pairs. ")
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/32_5.yaml')
    parser.add_argument('--output', type=str, default="exp/TAD")
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--only_test', action='store_true')

    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--device', type=int, default=0, help='GPU ID')

    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(data_type):

    args, config = parse_option()
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    logger.info(config)

    train_data, test_data, train_loader, test_loader = build_dataloader(logger, config)

    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
                          device="cpu", jit=False,
                          T=config.DATA.NUM_FRAMES,
                          droppath=config.MODEL.DROP_PATH_RATE,
                          use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                          use_cache=config.MODEL.FIX_TEXT,
                          logger=logger, )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if data_type == 'train data':
        # saved_path = './features/TAD/train_features_xclip.pkl'
        saved_path = './features/SHT/train_features_xclip.pkl'
        text_labels = generate_text(train_data)  # 从数据中生成文本标签

        start = time.time()
        get_features(model, train_loader, text_labels, device, saved_path, 'train', config)
        end = time.time()
        print(f'the processing time is: {end - start} s')

    elif data_type == 'test data':
        # saved_path = './features/TAD/test_features_xclip.pkl'
        saved_path = './features/SHT/test_features_xclip.pkl'
        text_labels = generate_text(train_data)
        start = time.time()
        get_features(model, test_loader, text_labels, device, saved_path, 'test', config)
        end = time.time()
        print(f'the processing time is: {end - start} s')

    elif data_type == 'video labels':  # for training data
        # saved_path = './features/TAD/video_level_labels.pkl'
        saved_path = './features/SHT/video_level_labels.pkl'
        start = time.time()
        get_video_labels(train_loader, device, saved_path)
        end = time.time()
        print(f'the processing time is {end - start} s')

    # elif data_type == 'video frame labels':  # for testing data
    #     saved_path_f = './features/TAD/gt_frame_labels.pkl'
    #     saved_path_v = './features/TAD/gt_video_labels.pkl'
    #     eval_path = './labels/TAD_test.txt'
    #     start = time.time()
    #     get_frame_video_labels(eval_path, saved_path_f, saved_path_v)
    #     end = time.time()
    #     print(f'the processing time is {end - start} s')

    else:
        raise ValueError('out of options')


if __name__ == '__main__':
    # main('test data')
    # main('train data')
    main('video labels')



