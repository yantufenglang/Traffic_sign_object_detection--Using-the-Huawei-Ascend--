import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
import os

def add_noise(image, noise_factor=0.1):
    """ 添加随机噪声 """
    noise = np.random.normal(0, noise_factor, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def augment_in_hsv(image):
    """ 在HSV空间中调整色彩 """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = cv2.subtract(s, random.randint(10, 30))  
    v = cv2.subtract(v, random.randint(20, 50))  

    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_in_ycrcb(image):
    """ 在YCrCb空间中调整亮度 """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y = cv2.subtract(y, random.randint(20, 50))

    ycrcb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def multi_channel_simulate_low_light(image_path, output_path):
    '''读取图像'''
    image = Image.open(image_path)


    brightness_factor = random.uniform(0.2, 0.7) 
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(brightness_factor)

    open_cv_image = np.array(brightened_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    lab = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    a = cv2.subtract(a, random.randint(10, 20))
    b = cv2.subtract(b, random.randint(10, 20))
    
    lab_image = cv2.merge((cl, a, b))
    enhanced_image_lab = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    enhanced_image_hsv = augment_in_hsv(enhanced_image_lab)
    enhanced_image_ycrcb = augment_in_ycrcb(enhanced_image_hsv)

    final_image = add_noise(enhanced_image_ycrcb, noise_factor=0.1)

    cv2.imwrite(output_path, final_image)

def process_folder(input_folder, output_folder):
    """ 批量处理文件夹中的照片 """
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')): 
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '_low_light' + os.path.splitext(filename)[1]
            output_path = os.path.join(output_folder, output_filename)

            multi_channel_simulate_low_light(input_path, output_path)
            print(f"Processed {input_path} -> {output_path}")

input_folder = 'input_images'  # 输入文件夹路径
output_folder = 'output_images'  # 输出文件夹路径
process_folder(input_folder, output_folder)