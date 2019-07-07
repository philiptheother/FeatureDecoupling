## PASCAL VOC 2007 Detection with Fast R-CNN

### Requirements

* [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn)

### Usage

* Run Fast R-CNN with multi-scale training and single scale testing, see the file [main.sh](https://github.com/philiptheother/FeatureDecoupling/blob/master/caffe_voc_detection/main.sh):  
`./main.sh`
* The pre-trained model expects RGB images that their pixel values are normalized with the following mean RGB values `mean_rgb = [0.485, 0.456, 0.406]` and std RGB values `std_rgb = [0.229, 0.224, 0.225]`. Prior to normalization the range of the image values must be [0.0, 1.0].
