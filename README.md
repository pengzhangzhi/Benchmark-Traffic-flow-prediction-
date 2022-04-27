Note: This repo is adpoted from https://github.com/UNIMIBInside/Smart-Mobility-Prediction. 

Due to technical reasons, I did not fork their code. 
# Introduction
This repo provide the implementations of baselines in the field traffic flow prediction. 
Most of the code in this field is too out-of-date to run, so I use docker to save you from installing tedious frameworks and provide one-line command to run the whole models.
Before running, make sure copy TaxiBJ dataset to the `data` folder.
Check Out `QuickStart`, where I provide out-of-the-box tutorial for you to use this repo!

## Install tedious frameworks with few lines of code
```
git clone https://github.com/pengzhangzhi/Benchmark-Traffic-flow-prediction-.git
cd Benchmark-Traffic-flow-prediction-
docker pull tensorflow/tensorflow:2.4.3-gpu
docker run -it tensorflow/tensorflow:2.4.3-gpu
pip install -r requirements.txt
```
## Run Baselines
```
bash train_TaxiBJ.sh
bash train_TaxiNYC.sh
```


## Repository structure
Each of the main folders is dedicated to a specific deep learning network. Some of them were taken and modified from other repositories associated with the source paper, while others are our original implementations. Here it is an exhaustive list:
* **ST-ResNet.** Folder for [[1]](#1). The original source code is [here](https://github.com/amirkhango/DeepST).
* **MST3D.** Folder with our original implementation of the model described in [[2]](#2).
* **Pred-CNN.** Folder for [[3]](#3). The original repository is [here](https://github.com/xzr12/PredCNN).
* **ST3DNet.** Folder for [[4]](#4). The starting-point code can be found [here](https://github.com/guoshnBJTU/ST3DNet).
* **STAR.** Folder for [[5]](#5). Soure code was taken from [here](https://github.com/hongnianwang/STAR).
* **3D-CLoST.** Folder dedicated to a model created during another research at Università Bicocca.
* **STDN.** Folder referring to [[6]](#6). This folder is actually a copy of [this](https://github.com/tangxianfeng/STDN) repository, since it was never used in our experimentes.
* **Autoencoder.** Refer to paper: Listening to the city, attentively: A Spatio-TemporalAttention Boosted Autoencoder for the Short-Term Flow Prediction Problem.

The contents of these folders can be a little different from each other, accordingly to the structure of the source repositories. Nevertheless, in each of them there are all the codes used to create input flow volumes, training and testing the models for single step prediction, and to evaluate performance on multi step prediction and transfer learning experiments.

The remaining folders are:
* **baselines**. Contains the code implementing Historical Average and ARIMA approaches to the traffic flow prediction problem.
* **data**. Folder where source data should be put in.
* **helpers**. Contains some helpers code used for data visualization or to get weather info through an external API.


## References
<a id="1">[1]</a> 
Zhang, Junbo, Yu Zheng, and Dekang Qi. "Deep spatio-temporal residual networks for citywide crowd flows prediction." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.

<a id="2">[2]</a>
Chen, Cen, et al. "Exploiting spatio-temporal correlations with multiple 3d convolutional neural networks for citywide vehicle flow prediction." 2018 IEEE international conference on data mining (ICDM). IEEE, 2018.

<a id="3">[3]</a>
Xu, Ziru, et al. "PredCNN: Predictive Learning with Cascade Convolutions." IJCAI. 2018.

<a id="4">[4]</a>
Guo, Shengnan, et al. "Deep spatial–temporal 3D convolutional neural networks for traffic data forecasting." IEEE Transactions on Intelligent Transportation Systems 20.10 (2019): 3913-3926.

<a id="5">[5]</a>
Wang, Hongnian, and Han Su. "STAR: A concise deep learning framework for citywide human mobility prediction." 2019 20th IEEE International Conference on Mobile Data Management (MDM). IEEE, 2019.

<a id="6">[6]</a>
Yao, Huaxiu, et al. "Revisiting spatial-temporal similarity: A deep learning framework for traffic prediction." Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019.

<a id="7">[7]</a>
Liu, Yang, et al. "Attention-based deep ensemble net for large-scale online taxi-hailing demand prediction." IEEE Transactions on Intelligent Transportation Systems 21.11 (2019): 4798-4807.

<a id="8">[8]</a>
Woo, Sanghyun, et al. "Cbam: Convolutional block attention module." Proceedings of the European conference on computer vision (ECCV). 2018.
