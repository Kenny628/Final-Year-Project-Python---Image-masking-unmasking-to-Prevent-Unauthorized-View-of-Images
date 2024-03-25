
import cv2
import numpy as np
import sys
import time
import math
from numpy import fft
from Crypto.Cipher import AES, Blowfish
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from flask import Flask, request, render_template, send_file, session
import multiprocessing
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import os
import random
import string
import uuid
import datetime
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from stegano import lsb
import numpy as np
from numpy import fft
import math
import cv2
from PIL import Image 
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
ivSize = AES.block_size
app = Flask(__name__)

app.secret_key = 'your_secret_key'
app.config['TEMPLATES_AUTO_RELOAD'] = True
UPLOAD_FOLDER = 'UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
string_data = "Hello, World!"
encoding_type = "utf-8"  # You can choose the encoding type you want

# Encoding the string to bytes
bytes_data = string_data.encode(encoding_type)

def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(17):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()             # 对点扩散函数进行归一化亮度
    else:
        for i in range(17):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()

# 对图片进行运动模糊
def make_blurred(input, PSF):
    input_fft = fft.fft2(input)             
    PSF_fft = fft.fft2(PSF) 
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def inverse(input, PSF):            
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF)           
    result = fft.ifft2(input_fft / PSF_fft) 
    result = np.abs(fft.fftshift(result))
    return result


def normal(array):
    array = np.where(array < 0,  0, array)
    array = np.where(array > 255, 255, array)
    array = array.astype(np.int16)
    return array

def is_32_bit_png(image_path):
    try:
        image = Image.open(image_path)
        if image.mode == "RGBA" and image.getextrema()[3] != (0, 255):
            return True
        return False
    except Exception as e:
        print("Error:", e)
        return False

def convert_32bit_to_24bit(input_path, output_path):
    # Open the 32-bit PNG image
    img = Image.open(input_path)
    
    # Ensure that the image has an alpha channel
    img = img.convert("RGBA")
    
    # Create a new image with 24-bit RGB format
    new_img = Image.new("RGB", img.size)
    
    # Paste the RGB content from the original image to the new image
    new_img.paste(img, (0, 0), img)
    
    # Save the new image as a 24-bit PNG
    new_img.save(output_path, format="PNG")

def generate_random_name(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def save_image_with_random_name(folder_path, image):
    while True:
        random_name = generate_random_name()
        image_path = os.path.join(folder_path, random_name + '.png')  # Change the extension as needed

        if not os.path.exists(image_path):
            cv2.imwrite(image_path, image)
            print(f"Image saved as '{random_name}.png'")
            return image_path
        else:
            print(f"File with name '{random_name}.png' already exists. Generating a new random name.")

def delete_expired_files():
    now = datetime.datetime.now()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            expiration_time = modification_time + datetime.timedelta(hours=1)
            if now > expiration_time:
                os.remove(file_path)
                print("deleted")
scheduler.add_job(delete_expired_files, 'interval', minutes=15)  # Run every 15 minutes
scheduler.start()
def keyCreation(key):
    block_size = 32
    # Getting a hash value of 256 bit (32 byte)
    key = hashlib.sha256(key.encode()).digest()
    return key
@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/imageDecrypt')
def imageDecrypt():
    return render_template('decrypt.html')    
@app.route('/encrypt', methods=['POST'])
def encrypt():
    start_time = time.time()
    image_file = request.files['image']
    password = request.form['password']
    key = keyCreation(password)
    iv = Random.new().read(AES.block_size)
    # using Cipher Block Chaining (CBC) Mode
    encryption_suite = AES.new(key, AES.MODE_CBC, iv)
    data = password.encode()
    block_size = 32
    # Encrypt the random initialize vector added with the padded data
    cipher_data = encryption_suite.encrypt(iv + pad(data, block_size))
    # Convert the cipher byte string to a base64 string to avoid decode padding error
    cipher_data = base64.b64encode(cipher_data).decode()
    print("Cipher text is :", cipher_data)

    imageUserInput=image_file.read()
    result = is_32_bit_png(image_file)
    if result:
        convert_32bit_to_24bit(image_file,'output_24bit.png')
        converted24bitImage=cv2.imread('output_24bit.png')
        # imageOrig = cv2.imdecode(np.frombuffer(blurred, np.uint8), cv2.IMREAD_COLOR)
        b_gray, g_gray, r_gray = cv2.split(converted24bitImage.copy())
    else:
        imageOrig = cv2.imdecode(np.frombuffer(imageUserInput, np.uint8), cv2.IMREAD_COLOR)
        b_gray, g_gray, r_gray = cv2.split(imageOrig.copy())

    img_h, img_w = b_gray.shape[:2]
    PSF = motion_process((img_h, img_w), 6000000000000000000000000000000000000000000000000)      # 进行运动模糊处理
    blurred_b = np.abs(make_blurred(b_gray, PSF))
    normal_blurred_b=normal(blurred_b)
    blurred_g = np.abs(make_blurred(g_gray, PSF))
    normal_blurred_g=normal(blurred_g)
    img_h, img_w = b_gray.shape[:2]
    blurred_r = np.abs(make_blurred(r_gray, PSF))
    normal_blurred_r=normal(blurred_r)
    blurred = cv2.merge([normal_blurred_b, normal_blurred_g, normal_blurred_r])

    filename = str(uuid.uuid4())
    filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename+".png")
    print(filepath)
    cv2.imwrite(filepath, blurred)
    secret = lsb.hide(filepath, cipher_data)
    filename = str(uuid.uuid4())
    filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename+".png")
    print(filepath)
    secret.save(filepath)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Encryption elapsed time:", elapsed_time, "seconds")
    return send_file(filepath, as_attachment=True,download_name="blurred_image.png")

@app.route('/get_session_data')
def get_session_data():
    user_name = session.get('user_name', 'Downloading')
    return user_name
@app.route('/decrypt', methods=['POST'])
def decrypt():
    session['username']=''
    error_message=''
    start_time = time.time()
    image_file = request.files['image']
    password = request.form['password']
    imageUserInput=image_file.read()
    imageOrig = cv2.imdecode(np.frombuffer(imageUserInput, np.uint8), cv2.IMREAD_COLOR)
    block_size = 32
    filename = str(uuid.uuid4())
    filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename+".png")
    print(filepath)
    # image_read=cv2.imread(filepath)
    cv2.imwrite(filepath, imageOrig)
    cipher_data = lsb.reveal(filepath)
    key = keyCreation(password)
    # if not cipher_data:
    #     return None

    cipher_data = base64.b64decode(cipher_data)
    # Retrieve the dynamic initialization vector saved
    iv = cipher_data[:AES.block_size]
    # Retrieved the cipher data
    cipher_data = cipher_data[AES.block_size:]
    decryption_suite = AES.new(key, AES.MODE_CBC, iv)
    try:
        decrypted_data = unpad(decryption_suite.decrypt(cipher_data),block_size)
        if(decrypted_data.decode('utf-8')==password):
            image2= cv2.imread(filepath)
            b_gray2, g_gray2, r_gray2 = cv2.split(image2.copy())

            img_h, img_w = b_gray2.shape[:2] 
            PSF = motion_process((img_h, img_w), 6000000000000000000000000000000000000000000000000)   
            result_blurred_b = inverse(b_gray2,PSF)  
            normal_deblurred_b=normal(result_blurred_b)
            img_h, img_w = g_gray2.shape[:2] 
            result_blurred_g = inverse(g_gray2,PSF)  
            normal_deblurred_g=normal(result_blurred_g)
            # PSF_3 =  extension_PSF(image, PSF_TEMP_3)  
            img_h, img_w = r_gray2.shape[:2] 
            result_blurred_r = inverse(r_gray2,PSF)
            normal_deblurred_r=normal(result_blurred_r)
            recovered = cv2.merge([normal_deblurred_b, normal_deblurred_g, normal_deblurred_r])
            # cv2.imwrite("recovere_lsb_aes_password.png",recovered)
        session['username']='Downloading'
    except ValueError as e:
        error_message = 'Invalid password. The provided password is incorrect.'
        # session['username'] = error_message
        session['username']=''
        return render_template('decrypt.html',error_message=error_message)
    # session.pop('username', None)
    session['username']='Downloading'
    filename = str(uuid.uuid4())
    filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename+".png")
    # expiration_time = datetime.datetime.now() + datetime.timedelta(hours=1)
    cv2.imwrite(filepath, recovered)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Decryption elapsed time:", elapsed_time, "seconds")
    return send_file(filepath, as_attachment=True,download_name="recovered.png")


if __name__ == '__main__':
   app.run()
