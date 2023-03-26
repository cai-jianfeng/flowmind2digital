# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: test_ocr.py
@Date: 2022/12/8 16:27
@Author: caijianfeng
"""
# In[1] easyocr

import easyocr
import cv2

reader = easyocr.Reader(['ch_sim', 'en'])
img = cv2.imread('ocr_test.jpg')
result = reader.readtext(img)
print(result)

# In[2] paddleocr

from paddleocr import PaddleOCR, draw_ocr
import time
import cv2

old_time = time.time()

# 图片地址
img_path = 'ocr_test.jpg'
ocr = PaddleOCR(lang="ch")  # 首次执行会自动下载模型文件，可以通过修改 lang 参数切换语种

result = ocr.ocr(img_path)
print(result)
