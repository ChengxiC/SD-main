import os
import torch
import torch.backends.cudnn as cudnn
import argparse
import datetime
import shutil
from pathlib import Path
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, evaluate_result
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
import mmcv
from utils.config import get_config
from models import xclip
from einops import rearrange
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


def parse_option():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/TAD.yaml')
    # parser.add_argument('--output', type=str, default="H:\\exp\\TAD")

    # parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/SDnormal.yaml')
    # parser.add_argument('--output', type=str, default="H:\\exp\\SDnormal")

    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/SDnormal.yaml')
    parser.add_argument('--output', type=str, default="H:\\SDnormal_baseline\\")

    parser.add_argument("--opts",   default=None, nargs='+',   help="Modify config options by adding 'KEY VALUE' pairs. ")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)

    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--w-smooth', default=0.01, type=float, help='weight of smooth loss')
    parser.add_argument('--w-sparse', default=0.001, type=float, help='weight of sparse loss')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--coffs', type=float, nargs='+', default=[1.0, 10.0])
    parser.add_argument('--k', type=int, default=3, help='number of top k')

    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(config):
    train_data, test_data, train_loader, test_loader = build_dataloader(logger, config)
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                          device="cpu", jit=False,
                          T=config.DATA.NUM_FRAMES,
                          droppath=config.MODEL.DROP_PATH_RATE,
                          use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                          use_cache=config.MODEL.FIX_TEXT,
                          logger=logger,)

    optimizer, _ = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    start_epoch, best_epoch, max_auc = 0, 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

    text_labels = generate_text(train_data)

    is_best = False

    if config.TEST.ONLY_TEST:
        if not os.path.isdir(config.MODEL.PRETRAINED):
            # evaluate on val set
            out_path = config.MODEL.PRETRAINED.replace('pth', 'pkl')
            if os.path.exists(out_path):
                scores_dict = mmcv.load(out_path)
            else:
                scores_dict = validate(test_loader, text_labels, model, config, out_path)

            tmp_dict = {}
            for v_name in scores_dict["prd"].keys():
                p_scores = np.array(scores_dict["prd"][v_name]).copy()
                if p_scores.shape[0] == 1:
                    tmp_dict[v_name] = [p_scores[0, :, 1]]
                else:
                    tmp_dict[v_name] = [p_scores[:, 0, 1]]

            auc_all, auc_ano = evaluate_result(tmp_dict, config.DATA.VAL_FILE)

            logger.info(f"AUC@all/ano of version {out_path.split('/')[-1]} on epoch {out_path.split('/')[-1].split('_')[-1][:-4]} : {auc_all:.4f}({auc_ano:.4f})")
            # logger.info(f"AUC@all/ano of version {out_path.split('/')[-2]} on epoch {out_path.split('/')[-1].split('_')[-1][:-4]} : {auc_all:.4f}({auc_ano:.4f})")
            return
        else:
            for epoch in range(config.TRAIN.EPOCHS):
                out_path = os.path.join(config.MODEL.PRETRAINED, 'ckpt_epoch_' + str(epoch) + '.pkl')
                scores_dict = validate(test_loader, text_labels, model, config, out_path)
                tmp_dict = {}
                for v_name in scores_dict["cls"].keys():
                    tmp_dict[v_name] = [np.array(scores_dict["prd"][v_name])[:, 0]]
                auc_all, auc_ano = evaluate_result(tmp_dict, config.DATA.VAL_FILE)

                is_best = auc_all > max_auc
                if is_best:
                    best_epoch = epoch
                max_auc = max(max_auc, auc_all)
                logger.info(f"Auc on epoch {epoch}: {auc_all:.4f}({auc_ano:.4f})")
                logger.info(f'Max AUC@all {best_epoch}/{epoch} : {max_auc:.4f}')

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_one_epoch(epoch, model, optimizer, lr_scheduler, train_loader, text_labels, config)

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):

            is_best = auc_all > max_auc if 'auc_all' in locals() else False
            epoch_saving(config, epoch, model, max_auc, optimizer, lr_scheduler, _, _, logger, config.OUTPUT, is_best)


def train_one_epoch(epoch, model, optimizer, lr_scheduler, train_loader, text_labels, config):

    model.train()
    optimizer.zero_grad()
    num_steps = len(train_loader)

    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    mil_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    texts = text_labels.cuda(non_blocking=True)
    k = args.k
    coff1 = args.coffs[0]
    coff2 = args.coffs[1]

    for idx, batch_data in enumerate(train_loader):

        images = batch_data["imgs"].cuda(non_blocking=True)[:, :1]
        label_id = batch_data["label"].cuda(non_blocking=True)[:, :1]
        label_id = label_id.reshape(-1)
        bz = images.shape[0]
        a_aug = images.shape[1]

        images = rearrange(images, 'b a k c t h w -> (b a k) t c h w')
        # b 是批次大小; a 是增强的数量; k 是剪辑帧的数量; c 是通道数; t 是时间维度或帧数; h 是图像的高度; w 是图像的宽度.

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)

        output = model(images, texts)
        logits = rearrange(output['y'], '(b a k) c -> (b a) k c', b=bz, a=a_aug,)
        scores = F.softmax(logits, dim=-1)
        scores_ano = scores[:, :, 1]
        scores_nor = scores[:, :, 0]

        normal_center_loss = torch.zeros(0)
        for j in range(bz):
            if label_id[j] == 0:
                normal_center = torch.mean(scores_ano[j], dim=-1)
                normal_center_loss = torch.cat((torch.mean((scores_ano[j] - normal_center.unsqueeze(-1)) ** 2, dim=-1).unsqueeze(0), normal_center_loss.to(scores.device)), dim=0)
            else:
                normal_center = torch.tensor([0.0], device=scores.device)
                normal_center_loss = torch.cat((normal_center, normal_center_loss.to(scores.device)), dim=0)
        normal_center_loss = normal_center_loss.sum() / bz

        topk_prob_ano, topk_indices_ano = torch.topk(scores_ano, k, dim=-1)
        topk_prob_nor, _ = torch.topk(scores_nor, k, dim=-1)

        logits_video = torch.gather(logits, 1, topk_indices_ano.unsqueeze(-1).expand(-1, -1, logits.size(-1))).mean(dim=1)
        topk_prob_video = torch.gather(scores, 1, topk_indices_ano.unsqueeze(-1).expand(-1, -1, logits.size(-1)))

        labels_binary = label_id > 0

        loss_mil = F.cross_entropy(logits_video, labels_binary.long(), reduction='none')
        loss_mil = loss_mil * topk_prob_video
        loss_mil = loss_mil.mean()

        total_loss = coff1 * loss_mil + coff2 * normal_center_loss
        # total_loss = loss_mil
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()

        total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        mil_loss_meter.update(loss_mil.item(), len(label_id))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch + 1}/{config.TRAIN.EPOCHS}][{idx + 20}/{num_steps}]\t'  
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'  
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'  
                f'tot {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'  
                f'mil {mil_loss_meter.val:.4f} ({mil_loss_meter.avg:.4f})\t'  
                f'gpu_mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(data_loader, text_labels, model, config, out_path):

    model.eval()
    vid_list = []

    anno_file = config.DATA.VAL_FILE

    with open(anno_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split()
            filename = line_split[0].split('/')[-1]
            vid_list.append(filename)

    start = time.time()  # 20241116
    total_clips = 0  # 20241116

    with torch.no_grad():

        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")

        scores_dict = dict()
        scores_dict['prd'] = dict()

        for idx, batch_data in enumerate(data_loader):
            _image = batch_data["imgs"]
            b, n, c, t, h, w = _image.size()
            total_clips += b   # 计算所有的clips # 20241116
            _image = rearrange(_image, 'b n c t h w -> (b n) t c h w')

            _image = _image.to('cuda')
            text_inputs = text_inputs.to('cuda')

            output = model(_image, text_inputs)

            scores_prd = F.softmax(output['y'], dim=-1)
            scores_prd = rearrange(scores_prd, '(b n) c -> b n c', b=b)
            scores_np_prd = scores_prd.cpu().data.numpy()

            for ind in range(scores_np_prd.shape[0]):
                v_name = vid_list[batch_data["vid"][ind]]
                if v_name not in scores_dict['prd']:
                    scores_dict['prd'][v_name] = []
                scores_dict['prd'][v_name].append(scores_np_prd[ind])

            if idx % 10 == 0:  # and len(data_loader) >= 100:
                logger.info(f'Test: [{idx}/{len(data_loader)}]\t')

    # 已经计算好scores # 20241116
    process_time = time.time() - start
    spc = process_time / total_clips
    logger.info(f'Test speed: {spc} s; total clips:{total_clips}; \t')  # 20241116

    tmp_dict = {}
    for v_name in scores_dict["prd"].keys():
        p_scores = np.array(scores_dict["prd"][v_name]).copy()
        if p_scores.shape[0] == 1:
            # 1,T,2
            tmp_dict[v_name] = [p_scores[0, :, 1]]
        else:
            # T,1,2
            tmp_dict[v_name] = [p_scores[:, 0, 1]]

    auc_all_p, auc_ano_p = evaluate_result(tmp_dict, config.DATA.VAL_FILE)

    logger.info(f'AUC: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\t')
    logger.info(f'writing results to {out_path}')
    mmcv.dump(scores_dict, out_path)
    return scores_dict


if __name__ == '__main__':

    torch.cuda.empty_cache()
    args, config = parse_option()
    torch.cuda.set_device(args.device)
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    logger.info(config)
    shutil.copy(args.config, config.OUTPUT)

    main(config)
