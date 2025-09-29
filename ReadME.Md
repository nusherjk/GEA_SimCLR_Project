#  Guided EvoNAS for Siamese Nets

### Research Project for University of Adelaide.

Manually designing neural networks with trial and error is a very expensive task in terms of  both time and resources. Which would have been plausible if the best performance was guaranteed. This is the same case for separate Neural Network model types such as Siamese Neural Networks(SNN) as well, SNN has multiple identical neural networks and does a comparison between the outputs which makes it more complex and thus making it more hard to manually design. However in recent years research in Neural Architecture Search methods has been quite popular for automating and optimizing neural networks.

Guided Evolutionary Neural Architecture Search for Siamese Network have been introduced and are being used to improve the performance and explore the complex search space of nearly infinite architecture configurations. These algorithms have their own merits and shortcomings as well. Siamese Networks are really useful in semi supervised learning tasks  such as representation learning, context based image retrieval, one shot image recognition. However designing those models efficiently is a a time consuming process which often does not result in success. The siamese networks consists of a an encoder and a projector head in each of the identical network. and even though different works have used tried and tested pre-trained models for encoder section the projection head part does have a heavy duty of releasing output embeddings which is ultimately does the feature comparison.


# Installation


``` 
pip install virtualenv
```

```
virtualenv venv 
```

this should create a virtualenv venv folder in the current working directory.

```
cd venv/Scirpts
```

```
activate
```

````
 pip install -r requirements.txt
````

Generate the MLP mini Bench configurations for Initialization.

```
py utils/genneraate_arch.py
```
this should create mlp_bench.json file. Finally run-- 

```
py main.py
```

for observation use tensorboard

```
    tensorboard --logdir=logs
```



