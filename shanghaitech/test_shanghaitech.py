import os
import numpy as np


def read_np_file(filepath):
    filename = os.path.basename(filepath).split('.')[0]
    try:
        np_file = np.load(filepath)
        return filename, np_file
    except FileNotFoundError:
        raise FileNotFoundError('check the path!!!')


def count_images(folder_path):

    image_count = 0
    image_extensions = ['.jpg', '.jpeg', '.png']

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1

    return image_count


def get_gt_1(filepath):

    filename, array = read_np_file(filepath)
    label = 1
    anomaly_start = -1
    anomaly_end = -1
    changes = 0

    is_anomaly = array[0] == 1

    for i, value in enumerate(array):
        if value == 1 and not is_anomaly:
            anomaly_start = i
            is_anomaly = True
            changes += 1  # 获取fluctuation 变化的数量
        elif value == 0 and is_anomaly:
            anomaly_end = i - 1
            is_anomaly = False
            changes += 1

    if is_anomaly:
        anomaly_end = len(array) - 1

    return filename, len(array), label, anomaly_start, anomaly_end, changes


def find_anomalies(array):

    anomalies = []
    in_anomaly = False
    start_index = -1

    for i in range(len(array)):
        if array[i] == 1 and not in_anomaly:
            start_index = i
            in_anomaly = True
        elif array[i] == 0 and in_anomaly:
            anomalies.append((start_index, i - 1))
            in_anomaly = False

    if in_anomaly:
        anomalies.append((start_index, len(array) - 1))

    return anomalies


def get_gt(filepath):
    filename, arr = read_np_file(filepath)
    label = 1
    total_frames = len(arr)
    lst = find_anomalies(arr)
    index = [ind for tup in lst for ind in tup]
    return f"{filename}, {label}, {total_frames}, "+", ".join(map(str, index))


def write_txt():
    anomaly_video_path = 'E:\\datasets\\shanghaitech\\testing\\test_frame_mask'
    test_label_path = 'shanghaitech_test.txt'

    with open(test_label_path, 'a') as f:
        for file in os.listdir(anomaly_video_path):
            filepath = os.path.join(anomaly_video_path, file)
            line = get_gt(filepath)
            f.write(line)
            f.write('\n')


def my_rename(datatype):
    if datatype == 'train':
        nor_frame_path = 'E:\\datasets\\shanghaitech_reorganized\\frames\\normal'
        for fl_name in os.listdir(nor_frame_path):
            old_name = os.path.join(nor_frame_path, fl_name)
            new_name = os.path.join(nor_frame_path, f'Normal_{fl_name}.mp4')
            os.rename(old_name, new_name)

    elif datatype == 'test':
        ano_frames_path = 'E:\\datasets\\shanghaitech_reorganized\\frames\\abnormal'
        txt_path = 'shanghaitech_anomaly.txt'
        filename_list = []
        ind = 0
        with open(txt_path, 'r') as file:
            for line in file:
                line = line.strip().split()  # 去除回车并按照空格进行分割
                filename = os.path.basename(str(line[0]))
                filename_list.append(filename)

        for fl_name in os.listdir(ano_frames_path):
            old_name = os.path.join(ano_frames_path, fl_name)
            new_name = os.path.join(ano_frames_path, filename_list[ind])
            os.rename(old_name, new_name)  # 替换名字
            ind += 1

    else:
        raise RuntimeError('out of options!!!')


def exchange_name_1_and_2():
    """
    coded by chengxi
    Returns:

    """
    filepath = 'shanghaitech_anomaly.txt'
    newpath = 'newtxt.txt'
    with open(filepath, 'r') as file:
        lines = file.readlines()
        print(lines)

    for i, line in enumerate(lines):
        if len(line) < 3:
            raise ValueError('the length is abnormal!!!')
        parts = line.strip().split()
        parts[1], parts[2] = parts[2], parts[1]
        # lines是list，没有.join()， 但lines[i]是str
        lines[i] = ' '.join(parts) + '\n'

    try:
        with open(newpath, 'w') as file:
            file.writelines(lines)
        print('finish writing!!!')

    except FileNotFoundError:
        raise FileNotFoundError(f'{newpath} not found')
    except Exception as e:
        print(f'open error!!! {e}')


def get_train_split(split_path, anomaly_path, normality_path, write_path, template_path):

    lst_5 = []
    lst_6 = []
    for line in open(split_path, 'r'):
        if len(line.strip().split('_')[1]) == 3:
            lst_5.append(line.strip('\n'))
        elif len(line.strip().split('_')[1]) == 4:
            lst_6.append(line.strip('\n'))

    for file in os.listdir(normality_path):
        file_path = os.path.join(normality_path, file)
        file_name = os.path.basename(file_path)
        for i, x in enumerate(lst_5):

            if file_name == 'Normal_' + x + '.mp4':
                parts = [None] * 4
                video_name = 'frames/normal/Normal_' + x + '.mp4'
                start_frameid = 0
                end_frameid = count_images(file_path) - 1
                label = 0  # normal data

                parts = [video_name, start_frameid, end_frameid, label]
                lst_5[i] = ' '.join(map(str, parts))

    print(lst_5)

    with open(template_path, 'r', encoding='utf-8') as file:
        template = file.readlines()
        print(len(template))
        # print(template)

    temp = []
    for file in os.listdir(anomaly_path):
        filepath = os.path.join(anomaly_path, file)
        filename = os.path.basename(filepath).split('.')[0]
        temp.append(filename)

    for i, item in enumerate(lst_6):
        part0 = item.strip('\n').split('.')[0].split('_')[0]
        part1 = item.strip('\n').split('.')[0].split('_')[-1]
        for line in temp:
            if part0 == line.split('_')[0] and part1 == line.split('_')[-1]:
                lst_6[i] = 'frames/abnormal/' + line + '.mp4'
                for t in template:  # 例如 'frames/abnormal/01_Cycling_0014.mp4 265 1 154 229\n'
                    if lst_6[i] == t.strip('\n').split(' ')[0]:
                        parts = [None] * 4
                        video_name = t.strip('\n').split(' ')[0]
                        start_frameid = 0
                        end_frameid = count_images('E:\\datasets\\shanghaitech_reorganized\\' + video_name)
                        label = int(os.path.basename(video_name).split('_')[0])
                        parts = [video_name, start_frameid, end_frameid, label]
                        lst_6[i] = ' '.join(map(str, parts))
    print(lst_6)

    # 最后写入.
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    with open(write_path, 'w') as file:
        for line in lst_6:
            file.write(line + '\n')

        for line in lst_5:
            file.write(line + '\n')


def get_test_split(split_path, anomaly_path, normality_path, write_path, template_path):
    """
    coded by chengxi
    Args:
        split_path: the given data split path, for example the weakly supervised split for shanghaitech data
        anomaly_path: the path of anomaly data
        normality_path: the path of normal data
        write_path: path to write the split
        template_path: the ground truth path for anomaly

    Returns:

    """

    lst_5 = []
    lst_6 = []
    for line in open(split_path, 'r'):
        if len(line.strip().split('_')[1]) == 3:
            lst_5.append(line.strip('\n'))
        elif len(line.strip().split('_')[1]) == 4:
            lst_6.append(line.strip('\n'))

    for file in os.listdir(normality_path):
        file_path = os.path.join(normality_path, file)
        file_name = os.path.basename(file_path)
        for i, x in enumerate(lst_5):

            if file_name == 'Normal_' + x + '.mp4':
                parts = [None] * 5
                video_name = 'frames/normal/Normal_' + x + '.mp4'
                start_frameid = 0
                end_frameid = count_images(file_path) - 1
                label = 0  # normal data

                parts = [video_name, end_frameid, label, start_frameid, end_frameid]
                lst_5[i] = ' '.join(map(str, parts))

    print(lst_5)

    with open(template_path, 'r', encoding='utf-8') as file:
        template = file.readlines()
        print(len(template))
        # print(template)

    temp = []
    for file in os.listdir(anomaly_path):
        filepath = os.path.join(anomaly_path, file)
        filename = os.path.basename(filepath).split('.')[0]
        temp.append(filename)

    for i, item in enumerate(lst_6):
        part0 = item.strip('\n').split('.')[0].split('_')[0]
        part1 = item.strip('\n').split('.')[0].split('_')[-1]
        for line in temp:
            if part0 == line.split('_')[0] and part1 == line.split('_')[-1]:
                lst_6[i] = 'frames/abnormal/' + line + '.mp4'
                for t in template:  # 例如 'frames/abnormal/01_Cycling_0014.mp4 265 1 154 229\n'
                    if lst_6[i] == t.strip('\n').split(' ')[0]:
                        lst_6[i] = t.strip('\n')
    print(lst_6)

    if not os.path.exists(write_path):
        os.makedirs(write_path)
    with open(write_path, 'w') as file:
        for line in lst_6:
            file.write(line + '\n')

        for line in lst_5:
            file.write(line + '\n')


if __name__ == '__main__':
    # my_rename('train')
    # my_rename('test')
    # exchange_name_1_and_2()

    test_path = 'test_split.txt'
    anomaly_path = 'E:\\datasets\\shanghaitech_reorganized\\frames\\abnormal'
    normality_path = 'E:\\datasets\\shanghaitech_reorganized\\frames\\normal'
    write_path_test = 'test1.txt'
    template_path = 'new_anomaly.txt'
    get_test_split(test_path, anomaly_path, normality_path, write_path_test, template_path)

    # train_path = 'train_split.txt'
    # anomaly_path = 'E:\\datasets\\shanghaitech_reorganized\\frames\\abnormal'
    # normality_path = 'E:\\datasets\\shanghaitech_reorganized\\frames\\normal'
    # write_path_test = 'train1.txt'
    # template_path = 'new_anomaly.txt'
    # get_train_split(train_path, anomaly_path, normality_path, write_path_test, template_path)













