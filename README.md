## *Self-Supervised Representation Learning by Rotation Feature Decoupling*

### Introduction

The current code implements on [pytorch](http://pytorch.org/) the following CVPR 2019 paper:  
**Title:**      "Self-Supervised Representation Learning by Rotation Feature Decoupling"  
**Authors:**     Zeyu Feng, Chang Xu, Dacheng Tao  
**Institution:** UBTECH Sydney AI Centre, School of Computer Science, FEIT, University of Sydney, Australia  
**Code:**        https://github.com/philiptheother/FeatureDecoupling  
**Link:**        [pdf and supp](http://openaccess.thecvf.com/content_CVPR_2019/html/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.html)

**Abstract:**  
We introduce a self-supervised learning method that focuses on beneficial properties of representation and their abilities in generalizing to real-world tasks. The method incorporates rotation invariance into the feature learning framework, one of many good and well-studied properties of visual representation, which is rarely appreciated or exploited by previous deep convolutional neural network based self-supervised representation learning methods. Specifically, our model learns a split representation that contains both rotation related and unrelated parts. We train neural networks by jointly predicting image rotations and discriminating individual instances. In particular, our model decouples the rotation discrimination from instance discrimination, which allows us to improve the rotation prediction by mitigating the influence of rotation label noise, as well as discriminate instances without regard to image rotations. The resulting feature has a better generalization ability for more various tasks. Experimental results show that our model outperforms current state-of-the-art methods on standard self-supervised feature learning benchmarks.

### Illustration

<img src="https://raw.githubusercontent.com/philiptheother/FeatureDecoupling/master/_imgs/figure.png">

### Citing FeatureDecoupling

If you find the code useful in your research, please consider citing our CVPR 2019 paper:
```
@InProceedings{Feng_2019_CVPR,
author = {Feng, Zeyu and Xu, Chang and Tao, Dacheng},
title = {Self-Supervised Representation Learning by Rotation Feature Decoupling},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

### License

Please refer to License files for details.

### Notice

* Inside the [FeatureDecoupling](https://github.com/philiptheother/FeatureDecoupling) directory with the downloaded code, the experiments-related data will be stored in the directories [_experiments](https://github.com/philiptheother/FeatureDecoupling/tree/master/_experiments).

* You have to make a copy of the file [config_env_example.py](https://github.com/philiptheother/FeatureDecoupling/blob/master/config_env_example.py) and rename it as *config_env.py*. Set in *config_env.py* the paths to where the caffe directory and datasets reside in your machine. 

### Experiments

* In order to train a FeatureDecoupling model in an unsupervised way with AlexNet-like architecture on the ImageNet training images and then evaluate linear classifiers (for ImageNet and Places205) as well as non-linear classifiers (for ImageNet) on top of the learned features please visit the [pytorch_feature_decoupling](https://github.com/philiptheother/FeatureDecoupling/tree/master/pytorch_feature_decoupling) folder.

### To do

* Pytorch-Caffe converter
* PASCAL VOC experiments
* Pre-trained models
