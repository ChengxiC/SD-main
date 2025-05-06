import os


def rename_images(folder_path):
    # 将 000.jpg 变为 0.jpg
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            name, ext = os.path.splitext(filename)
            new_name = str(int(name)) + ext
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_name)

            os.rename(old_file, new_file)


if __name__ == '__main__':
    file_path = 'E:\\datasets\\shanghaitech_reorganized\\frames\\abnormal'
    for dir in os.listdir(file_path):
        sub_file_path = os.path.join(file_path, dir)
        rename_images(sub_file_path)
        print(f'{sub_file_path} finished!')

    print('all finished!!!')



