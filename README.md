# Dyna-DM: Dynamic Object-aware Self-supervised Monocular Depth Maps


 >**Dyna-DM: Dynamic Object-aware Self-supervised Monocular Depth Maps**
 >
 >[[PDF](https://arxiv.org/pdf/2206.03799.pdf)]


<p align="center">
  <img src="./misc/arch.png"/>
</p>

## Install

The models were trained using CUDA 11.1, Python 3.7.x (conda environment), and PyTorch 1.7.0.

Create a conda environment with the PyTorch library:

```bash
conda create -n my_env python=3.7.4 pytorch=1.7.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda activate my_env
```

Install prerequisite packages listed in requirements.txt:

```bash
pip3 install -r requirements.txt
```

Also, ensure to install torch-scatter torch-sparse
```bash
pip3 install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
```

## Datasets

We use the datasets provided by [Insta-DM](https://github.com/SeokjuLee/Insta-DM) and evaluate the model with the [KITTI Eigen Split](https://arxiv.org/abs/1406.2283) using the raw [KITTI dataset](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip). 

## Models

Pretrained models for CityScape and KITTI+CityScape are provided here, where KITTI+CityScape is trained on both CityScape and KITTI and leads to the greatest depth estimations.

## Training

The models can be trained on the KITTI dataset by running:

```bash
bash scripts/train_kt.sh
```

Also, the models can be trained on the CityScape dataset by running:

```bash
bash scripts/train_cs.sh
```

The hyperparameters are defined in each script file and set at their defaults as stated in the paper.

## Evaluation

We evaluate the models by running:

```bash
bash scripts/run_eigen_test.sh
```

## References
 
* [Insta-DM](https://github.com/SeokjuLee/Insta-DM) (AAAI 2021, our baseline framework)

* [Struct2Depth](https://github.com/tensorflow/models/blob/archive/research/struct2depth) (AAAI 2019, object scale loss)

* [SC-SfMLearner](https://github.com/JiawangBian/SC-SfMLearner-Release) (NeurIPS 2019, our baseline framework)


 
