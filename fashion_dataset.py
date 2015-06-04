"""
 This script is aim to generating fashion datasets, which store the joint positions in xml files
 INPUTS:
 OUTPUTS:
"""

import xml.dom.minidom
import numpy as np
import glob
import os
import cv2 as cv

"""
 Configuration Fields
"""
# Directories
fashion_img_dir = '/Users/corsy/Documents/1/'
annota_file_dir = fashion_img_dir + 'annotation/'
output_files_dir = '/Users/corsy/Documents/1/outputs/'

# Parameters for generating training images
zoom_sclae = 1.7
displacement_offset_factor = 0.13
displacement_scale_factor = 0.14
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
def converge_to_onedir():
    return


"""
    Draw the joints according to joints position
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

    for j, joint in enumerate(joints):
        joint = (int(joint[0] * size[0]), int(joint[1] * size[1]))
        cv.circle(preview_img, joint, 5, (0, 0, 255), -1)

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

    joint_pos.extend(np.asarray(joint['lsho']))  # lsho
    joint_pos.extend(np.asarray(joint['head']))  # head
    joint_pos.extend(np.asarray(joint['rsho']))  # rsho

    return np.asarray(joint_pos).reshape((3, 2))

"""
    Mirror the target joints
"""
def mirror_joints(joint, size):

    joint_pos = []

    joint_pos.extend(np.asarray(joint['lsho']))  # lsho
    joint_pos.extend(np.asarray(joint['head']))  # head
    joint_pos.extend(np.asarray(joint['rsho']))  # rsho

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

        # Fetch all <AnnotationNode></AnnotationNode>
        bodies = element.getElementsByTagName('AnnotationNode')

        # Iterate the body to get the single joints
        for body_item in bodies:
                x_coordinates = body_item.getElementsByTagName('X')
                y_coordinates = body_item.getElementsByTagName('Y')
                visibilites = body_item.getElementsByTagName('Vis')
                joints = []

                print 'Body ', count
                # Get the values
                for i in range(0, 9, 1):
                    x = int(x_coordinates[i].firstChild.data)
                    y = int(y_coordinates[i].firstChild.data)
                    vis = int(visibilites[i].firstChild.data)
                    joints += (x, y)
                    print x, y, vis

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
                        img, joint_pos, box_ori, box_size, torso_box = crop_datas(img, joint_pos, torso_box, displacement, displacement_scale)

                        if img is None:
                            continue

                        # Save the image, bounding box and positions
                        cv.imwrite(output_files_dir + 'crop/m%s-%s.jpg' % (image_name, str(disp - 10)), img)
                        np.save(output_files_dir + 'joint/m%s-%s' % (image_name, str(disp - 10)), joint_pos)
                        train_list_file.write('m%s-%s' % (image_name, str(disp - 10))+'\n')
                        augment_count += 1

                        np.save(output_files_dir + 'ori/m%s-%s' % (image_name, str(disp - 10)), box_ori)
                        np.save(output_files_dir + 'size/m%s-%s' % (image_name, str(disp - 10)), box_size)
                        np.save(output_files_dir + 'torso/m%s-%s' % (image_name, str(disp)), torso_box)

                count += 1

                if count > 3:
                    break

        train_list_file.close()

if __name__ == '__main__':
    generate_datasets()