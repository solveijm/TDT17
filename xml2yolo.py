import xml.etree.ElementTree as ET
import glob
import os
import json
import cv2

# Inspired by: https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = float((bbox[0] * w) - w_half_len)
    ymin = float((bbox[1] * h) - h_half_len)
    xmax = float((bbox[0] * w) + w_half_len)
    ymax = float((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def xml_to_yolo(input_dir, output_dir, image_dir, classes):
    # create the labels folder (output directory)
    os.mkdir(output_dir)

    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(input_dir, '*.xml'))
    # loop through each 
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue

        result = []

        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            label = obj.find("name").text
            # If class is not a valid class, ignore remove the annotation
            if label not in classes:
                continue
            index = classes.index(label)
            pil_bbox = [float(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

    # generate the classes file as reference
    with open('classes.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(classes))


def formate_output(input_dir, output_file, image_dir):
    files = glob.glob(os.path.join(input_dir, '*.txt'))

    for file in files:
        basename = os.path.basename(file)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue
        image = cv2.imread(os.path.join(image_dir, f"{filename}.jpg"))
        height, width = image.shape[:2]
        result = []
        with open(file, 'r') as f1:
            lines = f1.readlines()
            for line in lines:
                elms = line.split(' ')
                result.append(elms[0])
                bb = yolo_to_xml_bbox([float(x) for x in elms[1:]], width, height)
                for el in bb:
                    result.append(el)
        with open(output_file, 'a+') as f2:
            f2.write(filename + " ")
            f2.write(" ".join(str(x) for x in result) + "\n")   
