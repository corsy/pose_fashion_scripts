#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 This script is aim to generating FLIC datasets
 INPUTS: FLIC directory
 OUTPUTS: FLIC training image list, preview image ....
"""

import cv2 as cv
import os
import glob
import numpy as np
from scipy.io import loadmat


# Configure datasets TODO[Luwei]: setup here
will_mirror_image = True
init_scale = 1.5
base_path = '/home/corsy/Documents/experiment/deeppose/datasets/FLIC/'             # Path to FLIC dataset
output_path = '/media/corsy/Experiments/triplejoints/'           # Path to outputs
displacement_offset_factor = 0.13
displacement_scale_factor = 0.14

# Configure properties of the datasets
crop_sizes = {
    '12-oc': (0, 0),
    'along': (0, 0),
    'batma': (0, 0),
    'bend-': (0, 0),
    'ten-c': (0, 0),
    'giant': (42, 396),
    'princ': (10, 464),
    'schin': (6, 461),
    'others': (56, 364)
}

joint_order = ['lsho', 'lelb',
               'lwri', 'rsho',
               'relb', 'rwri',
               'lhip','lkne',
               'lank', 'rhip',
               'rkne', 'rank',
               'leye', 'reye',
               'lear', 'rear',
               'nose', 'msho',
               'mhip', 'mear',
               'mtorso', 'mluarm',
               'mruarm', 'mllarm',
               'mrlarm', 'mluleg',
               'mruleg', 'mllleg',
               'mrlleg']

# Functions
def draw_joints_3(img, joints):
    size = img.shape
    preview_img = img

    sj = (int(joints[0, 0] * size[0]), int(joints[0, 1] * size[1]))
    nj = (int(joints[2, 0] * size[0]), int(joints[2, 1] * size[1]))
    cv.line(preview_img, sj, nj, (0, 255, 0), 3)

    sj2 = (int(joints[1, 0] * size[0]), int(joints[1, 1] * size[1]))
    nj2 = (int(((joints[2, 0] + joints[0, 0])/2) * size[0]),
           int(((joints[2, 1] + joints[0, 1])/2) * size[1]))
    cv.line(preview_img, sj2, nj2, (0, 255, 0), 3)

    # Draw specific joint
    index = 0
    joint = (int(joints[index, 0] * size[0]), int(joints[index, 1] * size[1]))
    cv.circle(preview_img, joint, 5, (0, 0, 255), -1)

def draw_joints(img, joints):
    size = img.shape
    preview_img = img

    for j, joint in enumerate(joints):
        if j != 2 and j != 3 and j + 1 < len(joints):
            sj = (int(joints[j, 0] * size[0]),
                  int(joints[j, 1] * size[1]))
            nj = (int(joints[j + 1, 0] * size[0]),
                  int(joints[j + 1, 1] * size[1]))
            cv.line(preview_img, sj, nj, (0, 255, 0), 3)

    for j, joint in enumerate(joints):
        joint = (int(joint[0] * size[0]), int(joint[1] * size[1]))
        cv.circle(preview_img, joint, 5, (0, 0, 255), -1)

    return preview_img

def get_joint_pos(joint):
    joint_pos = []

    # joint_pos.extend(np.asarray(joint['lwri']))  # lwri
    # joint_pos.extend(np.asarray(joint['lelb']))  # lelb
    joint_pos.extend(np.asarray(joint['lsho']))  # lsho

    head = np.asarray(joint['reye']) + \
        np.asarray(joint['leye']) + \
        np.asarray(joint['nose'])
    head /= 3

    joint_pos.extend(np.asarray(head))          # head
    joint_pos.extend(np.asarray(joint['rsho'])) # rsho
    # joint_pos.extend(np.asarray(joint['relb'])) # relb
    # joint_pos.extend(np.asarray(joint['rwri'])) # rwri

    return np.asarray(joint_pos).reshape((7, 2))

def update_joints_pos(joint, ori, size):
    joint -= np.array([ori[0], ori[1]])
    joint /= np.array([size[0], size[1]])

    return joint

def mirror_joints(joint, size):

    joint_pos = []

    # joint_pos.extend(np.asarray(joint['rwri']))  # lwri
    # joint_pos.extend(np.asarray(joint['relb']))  # lelb
    joint_pos.extend(np.asarray(joint['rsho']))  # lsho

    head = np.asarray(joint['reye']) + \
        np.asarray(joint['leye']) + \
        np.asarray(joint['nose'])
    head /= 3

    joint_pos.extend(np.asarray(head))          # head
    joint_pos.extend(np.asarray(joint['lsho'])) # rsho
    # joint_pos.extend(np.asarray(joint['lelb'])) # relb
    # joint_pos.extend(np.asarray(joint['lwri'])) # rwri

    joint_pos = np.asarray(joint_pos).reshape((3, 2))

    for i in range(0, 3):
        joint_pos[i][0] = size[1] - joint_pos[i][0]

    return joint_pos

def mirror_torso_box(joint, entry, size):

    if (np.isnan(np.all(rhip)) or np.isnan(np.all(lsho))):
        # Get frm entry
        box = entry['torsobox']
        box = np.array(box).reshape((2, 2))

        for i in range(0, 2):
            box[i][0] = size[1] - box[i][0]

        return box
    else:
        joint_pos = []
        joint_pos.extend(np.asarray(joint['rhip']))  # lelb
        joint_pos.extend(np.asarray(joint['lsho']))  # lsho
        joint_pos = np.asarray(joint_pos).reshape((2, 2))

        for i in range(0, 2):
            joint_pos[i][0] = size[1] - joint_pos[i][0]

        return joint_pos

def get_torso_box(joint, entry):

    if (np.isnan(np.all(rhip)) or np.isnan(np.all(lsho))):
        # Get frm entry
        box = entry['torsobox']
        box = np.array(box).reshape((2, 2))
        return box
    else:
        joint_pos = []
        joint_pos.extend(np.asarray(joint['rhip']))  # lelb
        joint_pos.extend(np.asarray(joint['lsho']))  # lsho
        joint_pos = np.asarray(joint_pos).reshape((2, 2))
        return joint_pos


def create_bbox(img_size, joint, scale, displacement, displacement_scale):

    min_x = int(np.min(joint[:, 0]))
    min_y = int(np.min(joint[:, 1]))
    max_x = int(np.max(joint[:, 0]))
    max_y = int(np.max(joint[:, 1]))

    width = (max_x - min_x)
    height = (max_y - min_y)

    if width/height < 0.8:
        width = 0.8*height
    elif height/width < 0.8:
        height = 0.8*width

    ext_width = width * (scale + displacement_scale)
    ext_height = height * (scale + displacement_scale)


    # Add displacement
    min_x += int(displacement[0] * width)
    min_y += int(displacement[1] * height)

    st_y = min_y + height / 2 - ext_height / 2
    st_y = st_y if st_y > 0 else 0

    en_y = min_y + height / 2 + ext_height / 2
    en_y = en_y if en_y < img_size[0] else img_size[0]

    st_x = min_x + width / 2 - ext_width / 2
    st_x = st_x if st_x > 0 else 0

    en_x = min_x + width / 2 + ext_width / 2
    en_x = en_x if en_x < img_size[1] else img_size[1]

    return (st_x, st_y), (en_x - st_x, en_y - st_y)


def crop_datas(img, joints, torso, displacement, displacement_scale):
    img_size = img.shape
    box_ori, box_size = create_bbox(img_size, joints, init_scale, displacement, displacement_scale)

    # Generate Joint position
    # Note that position have been normalized
    joints_pos = update_joints_pos(joints, box_ori, box_size)
    torso_pos = update_joints_pos(torso, box_ori, box_size)

    if np.all(joints_pos > 0):

        x = box_ori[0]
        y = box_ori[1]
        w = box_size[0]
        h = box_size[1]

        img = img[y:y+h, x:x+w, :]

        # Resize image to the 227x227 and save it
        img = cv.resize(img, (227, 227))

        return img, joints_pos, box_ori, box_size, torso_pos

    return None, None, None, None, None


def generate_sets():
    datasets = loadmat(base_path + 'examples.mat')
    datasets = datasets['examples'][0]

    joints_list = joint_order[:8]
    joints_list.extend(joint_order[12:14])
    joints_list.extend([joint_order[16]])

    if not os.path.exists(output_path + 'crop'):
        os.makedirs(output_path + 'crop')
    if not os.path.exists(output_path + 'ori'):
        os.makedirs(output_path + 'ori')
    if not os.path.exists(output_path + 'size'):
        os.makedirs(output_path + 'size')
    if not os.path.exists(output_path + 'joint'):
        os.makedirs(output_path + 'joint')
    if not os.path.exists(output_path + 'torso'):
        os.makedirs(output_path + 'torso')
    if not os.path.exists(output_path + 'preview'):
        os.makedirs(output_path + 'preview')

    train_list_file = open(output_path + 'trainlist.txt', 'w')
    test_list_file = open(output_path + 'testlist.txt', 'w')

    count = 0
    overall_train_count = 0
    train_count = 0
    test_count = 0

    for i, entry in enumerate(datasets):
        # Read the image file
        file_name = entry['filepath'][0]
        file_name = file_name.split('.')[0]

        # Should trained?
        istrain = entry['istrain'][0][0]
        if istrain == 0:
            test_list_file.write(file_name+'\n')
            test_count += 1
            continue

        train_count += 1

        # Iterate the entry in datasets
        ori_joint_pos = entry['coords'].T

        for disp in range(0, 20, 1):

            if disp < 10:
                joint_pos = np.copy(ori_joint_pos)
                joint_pos = dict(zip(joint_order, joint_pos))

                if not os.path.exists(base_path + 'images_c/%s.jpg' % file_name):
                    continue

                img = cv.imread(base_path + 'images_c/%s.jpg' % file_name)

                # Update the joints position correspond to exclude black zone
                for k, v in joint_pos.iteritems():
                    if np.all(~np.isnan(v)):
                        if file_name[:5] in crop_sizes.keys():
                            joint_pos[k][1] -= crop_sizes[file_name[:5]][0]
                        else:
                            joint_pos[k][1] -= crop_sizes['others'][0]

                torso_box = get_torso_box(joint_pos, entry)
                joint_pos = get_joint_pos(joint_pos)

                displacement_x = displacement_offset_factor * np.random.randn()
                displacement_y = displacement_offset_factor * np.random.randn()
                displacement = (displacement_x, displacement_y)

                displacement_scale = displacement_scale_factor * np.random.randn()

                # Crop the image and
                img, joint_pos, box_ori, box_size, torso_box = crop_datas(img, joint_pos, torso_box,
                                                                          displacement, displacement_scale)

                if img is None:
                    continue

                file_name = file_name.split('.')[0]
                count += 1

                # Save the image
                cv.imwrite(output_path + 'crop/%s-%s.jpg' % (file_name, str(disp)), img)
                np.save(output_path + 'joint/%s-%s' % (file_name, str(disp)), joint_pos)
                train_list_file.write('%s-%s' % (file_name, str(disp))+'\n')
                overall_train_count += 1

                np.save(output_path + 'ori/%s-%s' % (file_name, str(disp)), box_ori)
                np.save(output_path + 'size/%s-%s' % (file_name, str(disp)), box_size)
                np.save(output_path + 'torso/%s-%s' % (file_name, str(disp)), torso_box)

                # Draw the image
                preview_img = draw_joints(img, joint_pos)
                cv.imwrite(output_path + 'preview/%s-%s.jpg' % (file_name, str(disp)), preview_img)

            # If need mirror the image
            if disp >= 10 and will_mirror_image is True:

                joint_pos = np.copy(ori_joint_pos)
                joint_pos = dict(zip(joint_order, joint_pos))

                if not os.path.exists(base_path + 'images_c/%s.jpg' % file_name):
                    continue

                img = cv.imread(base_path + 'images_c/%s.jpg' % file_name)
                img = cv.flip(img, 1)

                # Update the joints position correspond to exclude black zone
                for k, v in joint_pos.iteritems():
                    if np.all(~np.isnan(v)):
                        if file_name[:5] in crop_sizes.keys():
                            joint_pos[k][1] -= crop_sizes[file_name[:5]][0]
                        else:
                            joint_pos[k][1] -= crop_sizes['others'][0]

                size = img.shape
                torso_box = mirror_torso_box(joint_pos, entry, size)
                joint_pos = mirror_joints(joint_pos, size)

                displacement_x = displacement_offset_factor * np.random.randn()
                displacement_y = displacement_offset_factor * np.random.randn()
                displacement = (displacement_x, displacement_y)

                displacement_scale = displacement_scale_factor * np.random.randn()

                # Crop the image and
                img, joint_pos, box_ori, box_size, torso_box = crop_datas(img, joint_pos, torso_box, displacement, displacement_scale)

                if img is None:
                    continue

                file_name = file_name.split('.')[0]
                count += 1

                # Save the image
                cv.imwrite(output_path + 'crop/m_%s-%s.jpg' % (file_name, str(disp - 40)), img)
                np.save(output_path + 'joint/m_%s-%s' % (file_name, str(disp - 40)), joint_pos)
                train_list_file.write('m_%s-%s' % (file_name, str(disp - 40))+'\n')
                overall_train_count += 1

                np.save(output_path + 'ori/m_%s-%s' % (file_name, str(disp - 40)), box_ori)
                np.save(output_path + 'size/m_%s-%s' % (file_name, str(disp - 40)), box_size)
                np.save(output_path + 'torso/m_%s-%s' % (file_name, str(disp - 40)), torso_box)

                # Draw the image
                preview_img = draw_joints(img, joint_pos)
                cv.imwrite(output_path + 'preview/m_%s-%s.jpg' % (file_name, str(disp - 40)), preview_img)

        if train_count % 20 == 0:
            print 'Now processing:', train_count


    train_list_file.close()
    test_list_file.close()

    print 'Total processed image:', count
    print 'Total training set:', train_count
    print 'Overall training set:', overall_train_count
    print 'Total test set:', test_count


def exclude_black_zone(basepath):
    if not os.path.exists(basepath + 'images_c'):
        os.mkdir(basepath + 'images_c')

    for fname in glob.glob(basepath +'images/*.jpg'):
        img = cv.imread(fname)
        pref = os.path.basename(fname)[:5]

        y = None
        h = None

        if pref in crop_sizes.keys():
            y, h = crop_sizes[pref]
            if h == 0:
                h = img.shape[0]
        else:
            y, h = crop_sizes['others']
        img = img[y:y + h, :, :]
        cv.imwrite(basepath + 'images_c/%s' % os.path.basename(fname), img)
        print fname

if __name__ == '__main__':
    # exclude_black_zone(work_path)
    generate_sets()