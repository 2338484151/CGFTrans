# Overview
A implementation for the paper "CGFTrans: Cross-modal Global Feature Fusion Transformer for Medical Report Generation"
## Environment:
python 3.6  
pytorch >= 1.7  
torchvision  
opencv-python  
## Datasets
We use two datasets (IU X-Ray and LGK) in our paper.  
For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.  
For `LGK`, come soon.  
## Train
You can run the file "train_iu_xray.sh" to train the project.  
## Test
You can use the file "main_test.py" to test the project. 
## Reference codes:
[https://github.com/cuhksz-nlp/R2Gen](https://github.com/cuhksz-nlp/R2Gen)https://github.com/cuhksz-nlp/R2Gen  
[https://github.com/cuhksz-nlp/R2GenCMN](https://github.com/cuhksz-nlp/R2GenCMN)https://github.com/cuhksz-nlp/R2GenCMN 
the evaluation metrics are from pycocoevalcap: https://github.com/tylin/coco-caption

