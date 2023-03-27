##### detection2安装教程

- 注意：需要先下 pytorch，不然会报错

- 下载：

```
git clone https://github.com/facebookresearch/detectron2.git
```



- 安装：

```
python -m pip install -e detectron2
```

- CPU运行示例(这个是示例用的，可以试一下)

```
python demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--input .\input.jpg \
--output .\output.jpg \
--opts MODEL.DEVICE cpu
```

- 运行检测代码(无arrow) --> 略

- 运行检测(有arrow)

  - 将数据集放在detectron2同级目录下(注意是放在第二个detectron2的同级目录，同级目录的还有以及存在的output文件夹)：

  ![1671285054390](C:\Users\86199\AppData\Roaming\Typora\typora-user-images\1671285054390.png)
  - 改数据集名字：将数据集改成 dataset_arrow，同时将里面的 flow_chart_new 改成 flow_chart，并将 config.txt 文件放入：

  ![1671285187192](C:\Users\86199\AppData\Roaming\Typora\typora-user-images\1671285187192.png)
  - 运行程序(须在第一个 detectron2 目录下)：

  ```python
  python train_keypoint.py
  ```