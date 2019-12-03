import cv2, os
from tqdm import tqdm

current_path = os.getcwd()
DATA_PATH = os.path.join(current_path, '../dataset')
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'faces_images/')

# data path
dataset_img_savepath_list = os.listdir(TRAIN_IMG_PATH)

# .png 파일만 조회
file_list_jpg = [file for file in dataset_img_savepath_list if file.endswith(".png")]
print ("file_list_png : {}".format(len(file_list_jpg)))

tbar = tqdm(range(len(file_list_jpg)), desc='\r')

for i in tbar:
    # get path of original and category images
    img_image_path = os.path.join(TRAIN_IMG_PATH, dataset_img_savepath_list[i])

    # open original and category images
    img_image = cv2.imread(img_image_path, cv2.IMREAD_COLOR)
    img_c, img_w, img_h = img_image.shape

    if img_h == 3:
        continue
    elif img_c == img_w == 128:
        continue
    else:
        print('This original image is broken : ', img_image_path)
print('Whole images are passed. No images are broken.')
