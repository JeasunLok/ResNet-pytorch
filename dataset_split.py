import os
import shutil
import random

input_folder = r'E:\202311\20231121class\Google dataset of SIRI-WHU_earth_im_tiff\12class_tif'
output_folder = r'dataset\SIRI-WHU'
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
image_cls = 200

output_folder_train = os.path.join(output_folder, "train")
output_folder_val = os.path.join(output_folder, "val")
output_folder_test = os.path.join(output_folder, "test")
if not os.path.exists(output_folder_train):
    os.mkdir(output_folder_train)
if not os.path.exists(output_folder_val):
    os.mkdir(output_folder_val)
if not os.path.exists(output_folder_test):
    os.mkdir(output_folder_test)

count = int(0)
image_list = []
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.split(".")[-1] == "tif":
            count = count + 1
            image_path = os.path.join(root, filename)
            cls = image_path.split("\\")[-2]
            image_list.append(image_path)
            random.shuffle(image_list)
            print(image_path, count, cls)
            if count%image_cls == 0:
                output_folder_cls_train = os.path.join(output_folder_train, cls)
                output_folder_cls_val = os.path.join(output_folder_val, cls)
                output_folder_cls_test = os.path.join(output_folder_test, cls)

                if not os.path.exists(output_folder_cls_train):
                    os.mkdir(output_folder_cls_train)
                if not os.path.exists(output_folder_cls_val):
                    os.mkdir(output_folder_cls_val)
                if not os.path.exists(output_folder_cls_test):
                    os.mkdir(output_folder_cls_test)

                train_image_list = image_list[0:int(image_cls*train_ratio)]
                val_image_list = image_list[int(image_cls*train_ratio):int(image_cls*(train_ratio+val_ratio))]
                test_image_list = image_list[int(image_cls*(train_ratio+val_ratio)):int(image_cls*(train_ratio+val_ratio+test_ratio))]

                for img_path in train_image_list:
                    shutil.copy(img_path, output_folder_cls_train)
                for img_path in val_image_list:
                    shutil.copy(img_path, output_folder_cls_val)
                for img_path in test_image_list:
                    shutil.copy(img_path, output_folder_cls_test)

                image_list = []