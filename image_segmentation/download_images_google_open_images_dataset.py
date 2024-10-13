import ast
import os
import shutil
import argparse
import zipfile
import time

import requests

import pandas as pd
import cv2


def mask_to_polygon(image_path, class_id):
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    ret = ''

    for polygon in polygons:
        for p_, p in enumerate(polygon):
            if p_ == len(polygon) - 1:
                ret = ret + '{}\n'.format(p)
            elif p_ == 0:
                ret = ret + str(class_id) + ' {} '.format(p)
            else:
                ret = ret + '{} '.format(p)
    return ret


def process(classes, data_out_dir):

    train_mask_data_url = 'https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv'
    val_mask_data_url = 'https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv'
    test_mask_data_url = 'https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv'

    downloader_url = 'https://raw.githubusercontent.com/openimages/dataset/master/downloader.py'
    class_names_all_url = 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv'
    class_ids_sem_seg_url = 'https://storage.googleapis.com/openimages/v7/oidv7-classes-segmentation.txt'

    for url in [train_mask_data_url, val_mask_data_url, test_mask_data_url, class_ids_sem_seg_url, class_names_all_url,
                downloader_url]:
        if not os.path.exists(url.split('/')[-1]):
            r = requests.get(url)
            with open(url.split('/')[-1], 'wb') as f:
                f.write(r.content)

    class_ids = []

    classes_all = pd.read_csv(class_names_all_url.split('/')[-1])

    with open(class_ids_sem_seg_url.split('/')[-1], 'r') as f:
        classes_sem_seg = [l[:-1] for l in f.readlines() if len(l) > 1]
        f.close()

    for class_ in classes:
        if class_ not in list(classes_all['DisplayName']):
            raise Exception('Class name not found: {}'.format(class_))
        class_index = list(classes_all['DisplayName']).index(class_)
        class_id_ = classes_all['LabelName'].iloc[class_index]
        if class_id_ in classes_sem_seg:
            class_ids.append(class_id_)
        else:
            raise Exception('Class name not found: {}'.format(class_))


    image_list_file_path = os.path.join('.', 'image_list_file')
    if os.path.exists(image_list_file_path):
        os.remove(image_list_file_path)

    image_list_file_list = []
    mask_paths = []
    for j, url in enumerate([train_mask_data_url, val_mask_data_url, test_mask_data_url]):
        filename = url.split('/')[-1]
        with open(filename, 'r') as f:
            line = f.readline()
            while len(line) != 0:
                mask_path, id, class_name, _, _, _, _, _, _, _ = line.split(',')[:13]
                if class_name in class_ids:
                    mask_paths.append(['train', 'validation', 'test'][j] + '/' + mask_path)
                    if id not in image_list_file_list:
                        image_list_file_list.append(id)
                        with open(image_list_file_path, 'a') as fw:
                            fw.write('{}/{}\n'.format(['train', 'validation', 'test'][j], id))
                            fw.close()
                line = f.readline()

            f.close()

    out_dir = './.out'
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    os.system('python downloader.py {} --download_folder={}'.format(image_list_file_path, out_dir))

    # download all masks
    out_masks_dir_ = './.out_masks_all'
    shutil.rmtree(out_masks_dir_, ignore_errors=True)
    os.makedirs(out_masks_dir_)
    for set_ in ['train', 'validation', 'test']:
        dir_ = os.path.join(out_masks_dir_, set_)
        # if os.path.exists(dir_):
        #     shutil.rmtree(dir_)
        os.makedirs(dir_, exist_ok=True)
        for k in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'a', 'b', 'c', 'd', 'e', 'f']:
            url = 'https://storage.googleapis.com/openimages/v5/{}-masks/{}-masks-{}.zip'.format(set_, set_, k)

            if not os.path.exists('{}-masks-{}.zip'.format(set_, k)):

                for retry in range(1, 10):
                    retry_because_of_timeout = False
                    try:
                        r = requests.get(url, timeout=5)
                    except Exception as e:
                        retry_because_of_timeout = True
                        print(e)

                    if retry_because_of_timeout:
                        print('retry', retry, url.split('/')[-1])
                        time.sleep(retry * 2 + 1)
                    else:
                        break

                with open(url.split('/')[-1], 'wb') as f:
                    f.write(r.content)

            with zipfile.ZipFile(url.split('/')[-1], 'r') as zip_ref:
                zip_ref.extractall(dir_ + '/')

            os.remove(url.split('/')[-1])

            for img_path_ in os.listdir(dir_):
                if '{}/{}'.format(set_, img_path_) not in mask_paths:
                    os.remove(os.path.join(dir_, img_path_))

    for set_ in ['train', 'validation', 'test']:
        for dir_ in [os.path.join(data_out_dir, 'images', set_),
                     os.path.join(data_out_dir, 'labels', set_)]:
            if os.path.exists(dir_):
                shutil.rmtree(dir_)
            os.makedirs(dir_)

    for mask_path in mask_paths:
        set_ = mask_path.split(os.sep)[0]
        image_id = mask_path.split(os.sep)[1][:-4][:16]
        label_name = mask_path.split(os.sep)[1][:-4][17:-9]
        print(mask_path, set_, image_id, label_name)
        if os.path.exists(os.path.join(out_dir, '{}.jpg'.format(image_id))):
            shutil.move(os.path.join(out_dir, '{}.jpg'.format(image_id)),
                        os.path.join(data_out_dir, 'images', set_, '{}.jpg'.format(image_id)))

        if os.path.exists(os.path.join(data_out_dir, 'images', set_, '{}.jpg'.format(image_id))):
            with open(os.path.join(data_out_dir, 'labels', set_, '{}.txt'.format(image_id)), 'a') as f:
                f.write('{}'.format(mask_to_polygon(os.path.join(out_masks_dir_, set_, mask_path.split(os.sep)[1]),
                                                    int([c.replace('/', '') for c in class_ids].index(label_name)))))
                f.close()

    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(out_masks_dir_, ignore_errors=True)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', default=['Duck'])
    parser.add_argument('--out-dir', default='./')
    args = parser.parse_args()

    classes = args.classes
    if type(classes) is str:
        classes = ast.literal_eval(classes)

    out_dir = args.out_dir

    process(classes, out_dir)
