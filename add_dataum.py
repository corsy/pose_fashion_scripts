#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lmdb
import shutil
import cv2 as cv
import os
import numpy as np
import caffe

"""
 Configuration Fields
"""
# FLIC TODO: Need to be modified manually
flic_base_path = '/home/corsy/Documents/experiment/deeppose/datasets/FLIC/histg/stage_2/lelb/'
flic_train_count = 269686

# FASHION TODO: Need to be modified manually
fashion_base_path = '/home/corsy/Documents/experiment/deeppose/datasets/FLIC/histg/stage_2/lelb/'
fashion_train_count = 269686

dataum_path = '/home/corsy/Documents/experiment/deeppose/datasets/FLIC/histg/stage_2/lelb/'
train_db_ext = 'train'
test_db_ext = 'test'


"""
 Functions
"""
def del_and_create(dname):
    if os.path.exists(dname):
        shutil.rmtree(dname)
    os.makedirs(dname)


def get_img_datum(image_fn):
    img = cv.imread(image_fn, cv.IMREAD_COLOR)
    img = img.swapaxes(0, 2).swapaxes(1, 2)
    datum = caffe.io.array_to_datum(img, 0)

    return datum


def get_jnt_datum(joint_fn):
    joint = np.load(joint_fn)
    joint = joint.flatten()

    datum = caffe.io.caffe_pb2.Datum()
    datum.channels = len(joint)
    datum.height = 1
    datum.width = 1
    datum.float_data.extend(joint.tolist())

    return datum

def create_dataset():
    # Initialize lmdb
    train_img_db_fn = dataum_path + 'image_'+train_db_ext +'.lmdb'
    del_and_create(train_img_db_fn)
    train_img_env = lmdb.Environment(train_img_db_fn, map_size=1099511627776)
    train_img_txn = train_img_env.begin(write=True, buffers=True)

    train_jnt_db_fn = dataum_path + 'joint_'+train_db_ext +'.lmdb'
    del_and_create(train_jnt_db_fn)
    train_jnt_env = lmdb.Environment(train_jnt_db_fn, map_size=1099511627776)
    train_jnt_txn = train_jnt_env.begin(write=True, buffers=True)

    # Initialize random keys
    keys = np.arange((flic_train_count + fashion_train_count))
    np.random.shuffle(keys)

    # Add FLIC files to dataum
    flic_train_file = open(flic_base_path + "suf_trainlist.txt")
    flic_train_img_path = flic_base_path + 'crop/'
    flic_train_jnt_path = flic_base_path + 'joint/'

    count = 0
    while 1:
        lines = flic_train_file.readlines(100000)
        if not lines:
            break
        for line in lines:
            size = len(line)
            line = line[0:size - 1]

            train_image_file = flic_train_img_path + line + '.jpg'
            train_joint_file = flic_train_jnt_path + line + '.npy'

            train_img_datum = get_img_datum(train_image_file)
            train_jnt_datum = get_jnt_datum(train_joint_file)
            key = '%010d' % keys[count]
            count += 1
            train_img_txn.put(key, train_img_datum.SerializeToString())
            train_jnt_txn.put(key, train_jnt_datum.SerializeToString())

            if count % 10000 == 0:
                train_img_txn.commit()
                train_jnt_txn.commit()
                train_img_txn = train_img_env.begin(write=True, buffers=True)
                train_jnt_txn = train_jnt_env.begin(write=True, buffers=True)

            if count % 100 == 0:
                print '[FLIC] Now processing:', count

    flic_train_file.close()

    # Add fashion sets to dataum
    fashion_train_file = open(fashion_base_path + "suf_trainlist.txt")
    fashion_train_img_path = fashion_base_path + 'crop/'
    fashion_train_jnt_path = fashion_base_path + 'joint/'

    count = 0
    while 1:
        lines = fashion_train_file.readlines(100000)
        if not lines:
            break
        for line in lines:
            size = len(line)
            line = line[0:size - 1]

            train_image_file = fashion_train_img_path + line + '.jpg'
            train_joint_file = fashion_train_jnt_path + line + '.npy'

            train_img_datum = get_img_datum(train_image_file)
            train_jnt_datum = get_jnt_datum(train_joint_file)
            key = '%010d' % keys[count]
            count += 1
            train_img_txn.put(key, train_img_datum.SerializeToString())
            train_jnt_txn.put(key, train_jnt_datum.SerializeToString())

            if count % 10000 == 0:
                train_img_txn.commit()
                train_jnt_txn.commit()
                train_img_txn = train_img_env.begin(write=True, buffers=True)
                train_jnt_txn = train_jnt_env.begin(write=True, buffers=True)

            if count % 100 == 0:
                print '[FASHION] Now processing:', count

    fashion_train_file.close()

    train_img_txn.commit()
    train_jnt_txn.commit()
    train_img_env.close()
    train_jnt_env.close()

def create_train_test_list():

    # Pharse the FLIC datasets
    flic_train_index = np.arange(flic_train_count)
    np.random.shuffle(flic_train_index)

    flic_train_file_array = ['None' for row in range(flic_train_count)]
    flic_train_list_file = open(flic_base_path+'trainlist.txt')

    count = 0
    while 1:
        lines = flic_train_list_file.readlines(100000)
        if not lines:
            break
        for line in lines:
            size = len(line)
            line = line[0:size - 1]
            img_name = line.split('/')[-1].split('.')[0]
            flic_train_file_array[flic_train_index[count]] = img_name
            count += 1

    suffle_train_list_file = open(flic_base_path + 'suf_trainlist.txt', 'w')
    for count, file_item in enumerate(flic_train_file_array):
        suffle_train_list_file.writelines(file_item+'\n')
    suffle_train_list_file.close()

    # Pharse the Fashion datasets
    fashion_train_index = np.arange(fashion_train_count)
    np.random.shuffle(fashion_train_index)

    fashion_train_file_array = ['None' for row in range(fashion_train_count)]
    fashion_train_list_file = open(fashion_base_path+'trainlist.txt')

    count = 0
    while 1:
        lines = fashion_train_list_file.readlines(100000)
        if not lines:
            break
        for line in lines:
            size = len(line)
            line = line[0:size - 1]
            img_name = line.split('/')[-1].split('.')[0]
            fashion_train_file_array[fashion_train_index[count]] = img_name
            count += 1

    suffle_train_list_file = open(fashion_base_path + 'suf_trainlist.txt', 'w')
    for count, file_item in enumerate(fashion_train_file_array):
        suffle_train_list_file.writelines(file_item+'\n')
    suffle_train_list_file.close()

if __name__ == '__main__':
    # Generate suffle train test first
    create_train_test_list()

    # Write to the databse
    create_dataset(flic_base_path)