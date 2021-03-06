# Depth-Preception
This file will state all packages neccesary and any conculsion made.
The model will be trained on 
https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html  
http://make3d.cs.cornell.edu/data.html  
https://www.a2d2.audi/a2d2/en/dataset.html  
https://lightfield-analysis.uni-konstanz.de/  
https://github.com/alexsax/taskonomy-sample-model-1   
https://www.cityscapes-dataset.com/login/   
http://www.cvlibs.net/datasets/kitti/eval_depth_all.php 
https://waymo.com/open/data/    
SLAM (Simultaneously Localization and Mapping)
In paper order:

Initial idea:
The netowrk knows representations of 3d objects in space (mainly from point clouds). We need accurate 3d model respesntations, from the 2d images of the object. We can start using the model by warping prevoius close represntations to the new image. Or we can start off with a plane for less xomplex shapes like the floor or walls. Our depth understanding comes from the knowledge of obhjects depth in 3d. The image must understand that a car is a 3d model, it is symmetrical and 3d. We would then be able to see wheere the car is in the 3d plane. We determine their depth as normal. USing sterio and optical flow, (with a few changes). Then we can predict the 3d models with more accruacy and predict the depth we cant see. creatinfg a 3d lmap of the scene with limited visual data. This is beneficial beciase it can outprefoem lidar and we can reconstruct a scene in 3d. Useful for VR and autonomus driving. And landing systems. 

