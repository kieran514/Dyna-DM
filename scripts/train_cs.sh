# Dataset directory
TRAIN_SET=datasets/CityScape/cityscapes_256


# Cityscapes model
PRETRAINED=checkpoints/pretrained/CS


# For training
CUDA_VISIBLE_DEVICES=0 python train.py $TRAIN_SET \
--pretrained-disp $PRETRAINED/resnet18_disp_cs.tar \
--pretrained-ego-pose $PRETRAINED/resnet18_ego_cs.tar \
--pretrained-obj-pose $PRETRAINED/resnet18_obj_cs.tar \
-b 1 -p 2.0 -c 1.0 -s 0.3 -o 0.02 -mc 0.1 -hp 0.2 -dm 0 -mni 20 -dmni 20 -objsmall 0.75 -maxtheta 0.9 \
--epoch-size 1000 \
--epochs 40 \
--with-ssim --with-mask --with-auto-mask \
--name final_cs \
--seed 42 
# --debug 


