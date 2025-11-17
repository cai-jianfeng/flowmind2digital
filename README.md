# flowmind2digital
We propose the Flowmind2digital method and the hdFlowmind dataset in this paper: Flowmind2digital is the first end-to-end recognition and conversion method for flowmind. The hdFlowmind is a dataset containing 1,776 hand-drawn and manually annotated flowminds, which considered a larger scope of 22 scenarios while having bigger quantity compared to previous works.Our experiments demonstrate the advantages of our method, whose accuracy on the hdFlowmind dataset is 87.3%, exceeding the previous SOTA work by 11.9%. At the same time, we also prove the effectiveness of our dataset. After pre-training our method and fine-tuning on Handwritten-diagram-dataset, the accuracy increased by 2.9%. Finally, we also discovered the importance of simple graphics for sketch recognition, that after adding simple data, the accuracy increased by 9.3%.

Our method is based on Detectron2, whose code is available in 'https://github.com/facebookresearch/detectron2.git'.

The mode parameter is available in 'https://huggingface.co/caijanfeng/flowmind2digital'.

The public dataset(hdflowmind) is available in 'https://huggingface.co/datasets/caijanfeng/hdflowmind'.

# 📄 License

This repository includes components licensed under the Apache License 2.0.
