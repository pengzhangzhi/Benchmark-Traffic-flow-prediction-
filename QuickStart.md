## Tutorials
1. pull docker image 
```
docker pull tensorflow/tensorflow:2.4.3-gpu
```
run a docker container:
```
docker run -it --mount type=bind,source=/user/pengzhangzhi/traffic/baseline/Smart-Mobility-Prediction,target=/user/Smart tensorflow/tensorflow:2.4.3-gpu
```
source refers to the absolute path of current repo, target denotes the mapped path in docker container.
 
2. install python package 

go to the target path and run the following command.
```
pip install -r requirements.txt
```
3. run baselines
```
bash train_TaxiBJ.sh
bash train_TaxiNYC.sh
```

## Some notes about docker

1. `Ctrl+P+Q` hung up container (not terminate!)
2. `docker attach your_container_name` resume the container. 
The two commnad allows you to run your experiments silently.

A toturial you may find very helpful. 

https://yeasy.gitbook.io/docker_practice/
