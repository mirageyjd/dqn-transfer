import cv2
from PIL import Image
from os import walk
import re
import random
import shutil
import numpy as np


TENNIS_IMG_DIR = './tennis_img/'
PONG_IMG_DIR = './pong_img/'
OUTPUT_DIR = './temp/'


def rotate_pong_img(pong_img_dir):
    '''
    Rotate Pong images in pong_img_dir folder all by 90 degrees counter-clockwise
    '''
    f = []
    for (_, _, filenames) in walk(pong_img_dir):
        f = filenames
    print('Processing following images: ')
    print(f)
    for file in f:
        if not file.endswith('.png'):
            continue
        img = Image.open(pong_img_dir + file)
        transposed = img.transpose(Image.ROTATE_90)
        transposed.save(pong_img_dir + file, 'PNG')


def img_greyscale_normalize(img_src_dir, img_dst_dir, pong=True):
    '''
    Run image preprocessing in 2 steps:
    i)  Turn into greyscale image
    ii) Mean pixel image normalization
    '''
    f = []
    for (_, _, filenames) in walk(img_src_dir):
        f = filenames
    print('Processing following images: ')
    mean_matrix = np.zeros(cv2.imread(img_src_dir + f[0]).shape)
    for file in f:
        img = cv2.imread(img_src_dir + file)
        mean_matrix += img
    mean_matrix /= len(f)
    print(mean_matrix)
    cv2.imwrite('./temp.png', mean_matrix)
    for file in f:
        img = cv2.imread(img_src_dir + file)
        img = img.astype(float)
        img -= mean_matrix
        img[img > 255] = 255
        img[img < 0] = 0
        if pong:
            # Crop the top 30 pixels for pong score
            for i in range(30):
                img[i] = np.zeros((img.shape[1], 3))
        else:
            # Crop the top 40 pixels for tennis score
            # for i in range(40):
            #     img[i] = np.zeros((img.shape[1], 3))
            img = img[40:(img.shape[0] - 20), 30:(img.shape[1] - 30)]
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        if pong:
            # Rotate the image
            h, w = img.shape[0], img.shape[1]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
            img = cv2.warpAffine(img, M, (h, w)) 
        cv2.imwrite('./temp/' + file, img)


def random_sample_img(img_src_dir, img_dst_dir):
    f = []
    for (_, _, filenames) in walk(img_src_dir):
        f = filenames
    print(f)
    tennis_file = list(filter(lambda x: x.startswith('tennis'), f))
    pong_file = list(filter(lambda x: x.startswith('pong'), f))
    # tennis_move_file = random.sample(tennis_file, 2100)
    # pong_move_file = random.sample(pong_file, 2100)
    for file in tennis_file:
        shutil.move(img_src_dir + file, img_dst_dir + 'testA/' + file)
    for file in pong_file:
        shutil.move(img_src_dir + file, img_dst_dir + 'testB/' + file)
    # for file in move_file:
    #     shutil.move(img_src_dir + file, img_dst_dir + file)


def file_rename(img_src_dir, name_prefix):
    f = []
    for (_, _, filenames) in walk(img_src_dir):
        f = filenames
    print(f)
    f = sorted(f)
    for i in range(len(f)):
        shutil.move(img_src_dir + f[i], img_src_dir + name_prefix + str(i) + '.png')


def list_file(img_src_dir):
    f = []
    for (_, _, filenames) in walk(img_src_dir):
        f = filenames
    print(f)
    f = sorted(f)
    for i in range(len(f)):
        print('./' + f[i])


# rotate_pong_img(PONG_IMG_DIR)
# img_greyscale_normalize(TENNIS_IMG_DIR, OUTPUT_DIR, False)
# img_greyscale_normalize(PONG_IMG_DIR, OUTPUT_DIR)
# random_sample_img('./datasets/atari/testB/', './datasets/atari/trainB/')
# random_sample_img('./temp/', './datasets/atari/')
# file_rename('./datasets/atari/testA/', 'tennis_test_')
# file_rename('./datasets/atari/testB/', 'pong_test_')
# file_rename('./datasets/atari/trainA/', 'tennis_train_')
# file_rename('./datasets/atari/trainB/', 'pong_train_')
list_file('./datasets/atari/testA/')
print('----')
list_file('./datasets/atari/testB/')
print('----')
list_file('./datasets/atari/trainA/')
print('----')
list_file('./datasets/atari/trainB/')