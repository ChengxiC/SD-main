from utils.tools import postpress, roc_auc_score, roc_curve
import numpy as np


def evaluate(vid2abnormality, anno_file, root=''):

    LABEL_PATH = anno_file
    gt = []
    ans = []
    GT = []
    ANS = []
    videos = {}

    for video in open(LABEL_PATH):
        vid = video.strip().split(' ')[0]
        video_len = int(video.strip().split(' ')[1])
        sub_video_gt = np.zeros((video_len,), dtype=np.int8)
        anomaly_tuple = video.split(' ')[3:]
        for ind in range(len(anomaly_tuple) // 2):
            start = int(anomaly_tuple[2 * ind])
            end = int(anomaly_tuple[2 * ind + 1])
            if start > 0:
                sub_video_gt[start:end] = 1
        videos[vid] = sub_video_gt

    for vid in videos:
        if vid not in vid2abnormality.keys():
            print("The video %s is excluded on the result!" % vid)
            continue

        cur_ab = np.array(vid2abnormality[vid])
        if cur_ab.shape[0] == 1:
            cur_ab = cur_ab[0, :, ]
        else:
            cur_ab = cur_ab[:, 0, ]
        cur_gt = np.array(videos[vid])
        ratio = float(len(cur_gt)) / float(len(cur_ab))
        cur_ans = np.zeros_like(cur_gt, dtype='float32')
        for i in range(len(cur_ab)):
            b = int(i * ratio + 0.5)
            e = int((i + 1) * ratio + 0.5)
            cur_ans[b: e] = cur_ab[i]

        cur_ans = postpress(cur_ans, seg_size=32)

        if cur_gt.max() >= 1:
            gt.extend(cur_gt.tolist())
            ans.extend(cur_ans.tolist())

        GT.extend(cur_gt.tolist())
        ANS.extend(cur_ans.tolist())

    ret = roc_auc_score(gt, ans)
    Ret = roc_auc_score(GT, ANS)
    fpr, tpr, threshold = roc_curve(GT, ANS)

    if root != '':
        output_file = root + "AUC.npz"
        np.savez(output_file, fpr=fpr, tpr=tpr, thre=threshold)

    return Ret, ret


