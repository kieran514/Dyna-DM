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
The network knows representations of 3d objects in space (mainly from point clouds). We need accurate 3d model representations, from the 2d images of the object. We can start using the model by warping previous close representations to the new image. Or we can start with a plane for less complex shapes like the floor or walls. Our depth understanding comes from the knowledge of objects depth in 3d. The image must understand that a car is a 3d model, it is symmetrical and 3d. We would then be able to see where the car is in the 3d plane. We determine their depth as normal. Using stereo and optical flow, (with a few changes). Then we can predict the 3d models with more accuracy and predict the depth we cant see. creating a 3d map of the scene with limited visual data. This is beneficial because it can outperform lidar and we can reconstruct a scene in 3d. Useful for VR and autonomous driving. And landing systems.




