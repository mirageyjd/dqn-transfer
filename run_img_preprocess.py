import cv2
from PIL import Image
from os import walk
import re
import random
import shutil


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


def img_greyscale_normalize(img_src_dir, img_dst_dir):
    '''
    Run image preprocessing in 2 steps:
    i)  Turn into greyscale image
    ii) Mean pixel image normalization
    '''
    f = []
    for (_, _, filenames) in walk(img_src_dir):
        f = filenames
    print('Processing following images: ')
    print(f)
    for file in f:
        img = cv2.imread(img_src_dir + file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(img_dst_dir + file, img)


def random_sample_img(img_src_dir, img_dst_dir):
    f = []
    for (_, _, filenames) in walk(img_src_dir):
        f = filenames
    print(f)
    move_file = random.sample(f, 20)
    for file in move_file:
        shutil.move(img_src_dir + file, img_dst_dir + file)


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
# img_greyscale_normalize(TENNIS_IMG_DIR, OUTPUT_DIR)
# img_greyscale_normalize(PONG_IMG_DIR, OUTPUT_DIR)
# random_sample_img('./datasets/atari/testB/', './datasets/atari/trainB/')
# file_rename('./datasets/atari/testA/', 'tennis_test_')
# file_rename('./datasets/atari/testB/', 'pong_test_')
# file_rename('./datasets/atari/trainA/', 'tennis_train_')
# file_rename('./datasets/atari/trainB/', 'pong_train_')
# list_file('./datasets/atari/testA/')
# print('----')
# list_file('./datasets/atari/testB/')
# print('----')
# list_file('./datasets/atari/trainA/')
# print('----')
# list_file('./datasets/atari/trainB/')