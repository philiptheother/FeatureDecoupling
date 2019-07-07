python tools/train_net.py --gpu 0 --solver models/AlexNet/solver.prototxt --iters 200000 --weights ../_experiments_conversion/alexnet_cls.caffemodel --cfg experiments/cfgs/multiscale.yml |& tee -a output/train.log

python tools/test_net.py --gpu 0 --def models/AlexNet/test.prototxt --net output/multiscale/voc_2007_trainval/alexnet_fast_rcnn_multiscale_iter_200000.caffemodel |& tee -a output/test.log
