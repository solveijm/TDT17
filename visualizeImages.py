import cv2
from xml.etree import ElementTree
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def draw_images(image_file, image_folder, annotation_folder):

    img = cv2.imread(image_folder + image_file.split('.')[0] + '.jpg')
    infile_xml = open(annotation_folder + image_file.split('.')[0] + '.xml')
    tree = ElementTree.parse(infile_xml)
    root = tree.getroot()
    
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # put text
        cv2.putText(img,cls_name,(xmin,ymin-10),font,1,(0,255,0),2,cv2.LINE_AA)

        # draw bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0),3)
    return img

def show_images_wo_annot(path, figsize=(20,10), columns=5):
    images = []
    for img_path in glob.glob(path + '/*.jpg'):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=figsize)
    columns = columns
    for i, image in enumerate(images):
        plt.subplot((int(len(images) / columns + 1)), columns, i + 1)
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.imshow(image)