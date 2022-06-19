# Dataset directory
TRAIN_SET=datasets/KITTI/kitti_256


# KITTI model
PRETRAINED=checkpoints/pretrained/CS+KITTI


# For training
CUDA_VISIBLE_DEVICES=0 python train.py $TRAIN_SET \
--pretrained-disp $PRETRAINED/resnet18_disp_cs+kt.tar \
--pretrained-ego-pose $PRETRAINED/resnet18_ego_cs+kt.tar \
--pretrained-obj-pose $PRETRAINED/resnet18_obj_cs+kt.tar \
-b 1 -p 2.0 -c 1.0 -s 0.3 -o 0.02 -mc 0.1 -hp 0.2 -dm 0 -mni 20 -dmni 20 -objsmall 0.75 -maxtheta 0.9 \
--epoch-size 1000 \
--epochs 40 \
--with-ssim --with-mask --with-auto-mask \
--with-gt \
--name final_kt \
--seed 42 
# --debug \

