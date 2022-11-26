from xml.etree import ElementTree
import os

class Country_stats:
    def __init__(self, country_name, images_per_class, total_images, total_images_with_detection):
        self.country_name = country_name
        self.images_per_class = images_per_class
        self.total_images = total_images
        self.total_images_with_detection = total_images_with_detection
        self.number_of_labels = sum(images_per_class.values())

def create_imageSet_folder(imageSetsPath):
    # Create folder to save which classes that can be found in which image
    isExist = os.path.exists(imageSetsPath)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(imageSetsPath)
        print("The new directory is created!")
    else: # Remove the existing files in the folder
        for file in os.listdir(imageSetsPath):
            path = os.path.join(imageSetsPath,file)
            os.remove(path)

def get_stats(src_dir_annot, country_name, imageSetsPath):
    create_imageSet_folder(imageSetsPath)
    # the number of total images and total labels.
    images_per_class= {}
    total_images = 0
    total_images_with_detection = 0
        
    file_list = [filename for filename in os.listdir(src_dir_annot) if not filename.startswith('.')]

    for file in file_list:
        total_images = total_images + 1
        if file =='.DS_Store':
            pass
        else:
            infile_xml = open(src_dir_annot + file)
            tree = ElementTree.parse(infile_xml)
            root = tree.getroot()
            image_path = root.find('filename').text.strip('.jpg')

            crack_in_image = False
            for obj in root.iter('object'):
                crack_in_image = True
                cls_name = obj.find('name').text
                path = imageSetsPath + f'/{cls_name}_images.txt'
                with open(path, 'a+') as f:
                    f.write(f'{image_path}\n')
                if cls_name in images_per_class.keys():
                    images_per_class[cls_name] += 1
                else:
                    images_per_class[cls_name] = 1
            if crack_in_image:
                total_images_with_detection += 1
    return Country_stats(country_name, images_per_class, total_images, total_images_with_detection)