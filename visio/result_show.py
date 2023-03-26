# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: result_show.py
@Date: 2022/12/18 18:57
@Author: caijianfeng
"""
import json

with open('./output/metrics.json', encoding='utf-8') as f:
    metrics = json.load(f)
print(metrics)