#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 This script is aim to generating fashion datasets, which store the joint positions in xml files
 INPUTS: Fashion datasets directory
 OUTPUTS: Fashion training image lists, preview images.
"""

import xml.dom.minidom
import numpy as np
import glob
import os
import cv2 as cv
import shutil

"""
 Configuration Fields
"""

# TODO[Luwei]: setup configuration here

# Directories
fashion_img_dir = '/Users/corsy/Documents/1/'
annota_file_dir = fashion_img_dir + 'annotation/'
output_files_dir = '/Users/corsy/Documents/1/outputs/'

# Parameters for generating training images
zoom_sclae = 3.8
displacement_offset_factor = 0.12
displacement_scale_factor = 0.1
will_mirror_image = True

# Joint order configuration in Fashion datasets
joint_order = ['head',
               'lsho', 'rsho',
               'lelb', 'relb',
               'lwri', 'rwri',
               'lhip', 'rhip']


"""
 Functions for generating training image lists.
"""

# TODO[Luwei]: will implement this method later
def merge_to_onedir():

    # TODO[Luwei]: setup distribute directory
    distribute_dir = ''

    for i in range(1, 12, 1):
        current_dir = distribute_dir + str(i) + '/anno'
        files = glob.glob(current_dir + '*.xml')

        # Copy to 'annota_file_dir'
        for file_item in files:
            shutil.copy(file_item, annota_file_dir+file.split('/')[-1])

    return


"""
    Draw the joints according to joints position (7-joints)
    INPUTS: image that will be drawn, joint positions
    OUTPUTS: marked image
"""
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

"""
    Draw the joints according to joints position (7-joints)
    INPUTS: image that will be drawn, joint positions
    OUTPUTS: marked image
"""
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

    sj = (int(joints[2, 0] * size[0]),
          int(joints[2, 1] * size[1]))
    nj = (int(joints[4, 0] * size[0]),
          int(joints[4, 1] * size[1]))
    cv.line(preview_img, sj, nj, (0, 255, 0), 3)

    sj = ((sj[0] + nj[0]) / 2,
          (sj[1] + nj[1]) / 2)
    nj = (int(joints[3, 0] * size[0]),
          int(joints[3, 1] * size[1]))
    cv.line(preview_img, sj, nj, (0, 255, 0), 3)

    # Draw specific joint
    index = 2
    joint = (int(joints[index, 0] * size[0]), int(joints[index, 1] * size[1]))
    cv.circle(preview_img, joint, 5, (0, 0, 255), -1)

    # for j, joint in enumerate(joints):
    #     joint = (int(joint[0] * size[0]), int(joint[1] * size[1]))
    #     cv.circle(preview_img, joint, 5, (0, 0, 255), -1)


    # sj = (int(joints[2, 0] * size[0]),
    #       int(joints[2, 1] * size[1]))
    # nj = (int(joints[4, 0] * size[0]),
    #       int(joints[4, 1] * size[1]))
    # cv.circle(preview_img, ((sj[0] + nj[0]) / 2,
    #                         (sj[1] + nj[1]) / 2),
    #                          5, (0, 0, 255), -1)

    return preview_img

"""
    Extract the torso
"""
def get_torso_box(joint):
    joint_pos = []
    joint_pos.extend(np.asarray(joint['rhip']))  # lelb
    joint_pos.extend(np.asarray(joint['lsho']))  # lsho
    joint_pos = np.asarray(joint_pos).reshape((2, 2))
    return joint_pos

"""
    Extract the torso for mirroring
"""
def mirror_torso_box(joint, size):
    joint_pos = []
    joint_pos.extend(np.asarray(joint['rhip']))  # lelb
    joint_pos.extend(np.asarray(joint['lsho']))  # lsho
    joint_pos = np.asarray(joint_pos).reshape((2, 2))

    for i in range(0, 2):
        joint_pos[i][0] = size[1] - joint_pos[i][0]

    return joint_pos


"""
    Extract the target joints
"""
def get_joint_pos(joint):
    joint_pos = []

    # joint_pos.extend(np.asarray(joint['rwri']))  # lwri
    # joint_pos.extend(np.asarray(joint['relb']))  # lelb
    joint_pos.extend(np.asarray(joint['lsho']))  # lsho
    joint_pos.extend(np.asarray(joint['head']))  # head
    joint_pos.extend(np.asarray(joint['rsho']))  # rsho
    # joint_pos.extend(np.asarray(joint['lelb']))  # relb
    # joint_pos.extend(np.asarray(joint['lwri']))  # rwri

    return np.asarray(joint_pos).reshape((3, 2))

"""
    Mirror the target joints
"""
def mirror_joints(joint, size):

    joint_pos = []

    # joint_pos.extend(np.asarray(joint['lwri']))  # lwri
    # joint_pos.extend(np.asarray(joint['lelb']))  # lelb
    joint_pos.extend(np.asarray(joint['rsho']))  # lsho
    joint_pos.extend(np.asarray(joint['head']))  # head
    joint_pos.extend(np.asarray(joint['lsho']))  # rsho
    # joint_pos.extend(np.asarray(joint['relb']))  # relb
    # joint_pos.extend(np.asarray(joint['rwri']))  # rwri

    joint_pos = np.asarray(joint_pos).reshape((3, 2))

    for i in range(0, 3):
        joint_pos[i][0] = size[1] - joint_pos[i][0]

    return joint_pos


"""
    Create the bounding box based on joint positions, zooming scales and displacments
    INPUTS: size of image, joint position, scaling, displacements (translation + scaling)
    OUTPUTS: bounding box original point and size
"""
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


"""
    Update the joint positions according to the bounding box
    INPUT: original point of bounding box, size of the box
    OUTPUT: updated joint positions
"""
def update_joints_pos(joint, ori, size):
    joint -= np.array([ori[0], ori[1]])
    joint /= np.array([size[0], size[1]])

    return joint


"""
    Crop the images based on joints
    INPUT: image, joint positions, torso box, translation displacement, scale displacement
    OUTPUT: cropped image, cropped positions, bounding box location and size
"""
def crop_datas(img, joints, torso,  displacement, displacement_scale):
    img_size = img.shape
    box_ori, box_size = create_bbox(img_size, joints, zoom_sclae, displacement, displacement_scale)

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


"""
    Main routine for generating training images
"""
def generate_datasets():
    bodies_config_files = glob.glob(annota_file_dir + '*.xml')

    if not os.path.exists(output_files_dir + 'crop'):
        os.makedirs(output_files_dir + 'crop')
    if not os.path.exists(output_files_dir + 'ori'):
        os.makedirs(output_files_dir + 'ori')
    if not os.path.exists(output_files_dir + 'size'):
        os.makedirs(output_files_dir + 'size')
    if not os.path.exists(output_files_dir + 'joint'):
        os.makedirs(output_files_dir + 'joint')
    if not os.path.exists(output_files_dir + 'torso'):
        os.makedirs(output_files_dir + 'torso')
    if not os.path.exists(output_files_dir + 'preview'):
        os.makedirs(output_files_dir + 'preview')

    train_list_file = open(output_files_dir + 'trainlist.txt', 'w')

    count = 0   # Count how many body will be processed
    augment_count = 0 # Augmented images count
    for body_file_path in bodies_config_files:

        # Get the xml dom
        dom = xml.dom.minidom.parse(body_file_path)
        element = dom.documentElement

        # Get the image name
        image_name = element.getElementsByTagName('ImageName')[0].firstChild.data.split('.')[0]
        ori_img = cv.imread(fashion_img_dir + image_name + '.jpg')
        print image_name

        # Fetch all <AnnotationNode></AnnotationNode>
        bodies = element.getElementsByTagName('AnnotationNode')

        # Iterate the body to get the single joints
        for body_item in bodies:
                x_coordinates = body_item.getElementsByTagName('X')
                y_coordinates = body_item.getElementsByTagName('Y')
                visibilites = body_item.getElementsByTagName('Vis')
                joints = []

                # print 'Body ', count
                # Get the values
                for i in range(0, 9, 1):
                    x = float(x_coordinates[i].firstChild.data)
                    y = float(y_coordinates[i].firstChild.data)
                    vis = int(visibilites[i].firstChild.data)
                    joints += (x, y)
                    # print x, y, vis

                joints = np.asarray(joints).reshape((9, 2))

                # Now that come to the basic routine for joint generation
                # We need 2x * 10x per image, 2x for flip, 10x for random displacment + scaling
                for disp in range(0, 20, 1):
                    # normal
                    if disp < 10:
                        joint_pos = np.copy(joints)
                        joint_pos = dict(zip(joint_order, joint_pos))
                        img = ori_img[:].copy()

                        size = img.shape
                        torso_box = get_torso_box(joint_pos)
                        joint_pos = get_joint_pos(joint_pos)

                        # Displacements values
                        displacement_x = displacement_offset_factor * np.random.randn()
                        displacement_y = displacement_offset_factor * np.random.randn()
                        displacement = (displacement_x, displacement_y)
                        displacement_scale = displacement_scale_factor * np.random.randn()

                        # Crop the image and
                        img, joint_pos, box_ori, box_size, torso_box = crop_datas(img, joint_pos, torso_box,
                                                                                  displacement, displacement_scale)

                        if img is None:
                            continue

                        # Save the image, bounding box and positions
                        cv.imwrite(output_files_dir + 'crop/%s-%s.jpg' % (image_name, str(disp)), img)
                        np.save(output_files_dir + 'joint/%s-%s' % (image_name, str(disp)), joint_pos)
                        train_list_file.write('%s-%s' % (image_name, str(disp))+'\n')
                        augment_count += 1

                        np.save(output_files_dir + 'ori/%s-%s' % (image_name, str(disp)), box_ori)
                        np.save(output_files_dir + 'size/%s-%s' % (image_name, str(disp)), box_size)
                        np.save(output_files_dir + 'torso/%s-%s' % (image_name, str(disp)), torso_box)

                        draw_joints_3(img, joint_pos)
                        cv.imwrite(output_files_dir + 'preview/%s-%s.jpg' % (image_name, str(disp)), img)

                    # Mirroring
                    if disp >= 10 and will_mirror_image is True:
                        joint_pos = np.copy(joints)
                        joint_pos = dict(zip(joint_order, joint_pos))
                        img = ori_img[:].copy()
                        img = cv.flip(img, 1)

                        size = img.shape
                        torso_box = mirror_torso_box(joint_pos, size)
                        joint_pos = mirror_joints(joint_pos, size)

                        # Displacements values
                        displacement_x = displacement_offset_factor * np.random.randn()
                        displacement_y = displacement_offset_factor * np.random.randn()
                        displacement = (displacement_x, displacement_y)
                        displacement_scale = displacement_scale_factor * np.random.randn()

                        # Crop the image and
                        img, joint_pos, box_ori, box_size, torso_box = crop_datas(img, joint_pos, torso_box,
                                                                                  displacement, displacement_scale)

                        if img is None:
                            continue

                        # Save the image, bounding box and positions
                        cv.imwrite(output_files_dir + 'crop/m_%s-%s.jpg' % (image_name, str(disp - 10)), img)
                        np.save(output_files_dir + 'joint/m_%s-%s' % (image_name, str(disp - 10)), joint_pos)
                        train_list_file.write('m_%s-%s' % (image_name, str(disp - 10))+'\n')
                        augment_count += 1

                        np.save(output_files_dir + 'ori/m_%s-%s' % (image_name, str(disp - 10)), box_ori)
                        np.save(output_files_dir + 'size/m_%s-%s' % (image_name, str(disp - 10)), box_size)
                        np.save(output_files_dir + 'torso/m_%s-%s' % (image_name, str(disp - 10)), torso_box)
                        draw_joints_3(img, joint_pos)
                        cv.imwrite(output_files_dir + 'preview/m_%s-%s.jpg' % (image_name, str(disp - 10)), img)

        count += 1

        if count > 3:
            break

    train_list_file.close()

if __name__ == '__main__':
    # If need to merge all distribute fashion datasets to one direcotry
    merge_to_onedir()

    # Generate the fashion training list for later use
    generate_datasets()