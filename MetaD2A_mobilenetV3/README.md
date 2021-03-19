# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets
This code is for MobileNetV3 Search Space experiments


## Prerequisites
- Python 3.6 (Anaconda)
- PyTorch 1.6.0
- CUDA 10.2
- python-igraph==0.8.2
- tqdm==4.50.2
- torchvision==0.7.0
- python-igraph==0.8.2
- scipy==1.5.2
- ofa==0.0.4-2007200808


## MobileNetV3 Search Space
Go to the folder for MobileNetV3 experiments (i.e. ```MetaD2A_mobilenetV3```)

The overall flow is summarized as follows:
- Building database for Predictor
- Meta-Training Predictor
- Building database for Generator with trained Predictor
- Meta-Training Generator
- Meta-Testing (Searching)
- Evaluating the Searched architecture


## Data Preparation
To download preprocessed data files, run ```get_files/get_preprocessed_data.py```: 
```shell script
$ python get_files/get_preprocessed_data.py
```
It will take some time to download and preprocess each dataset.


## Meta Test and Evaluation
### Meta-Test

You can download trained checkpoint files for generator and predictor
```shell script
$ python get_files/get_generator_checkpoint.py
$ python get_files/get_predictor_checkpoint.py
```

If you want to meta-test with your own dataset, please first make your own preprocessed data, 
by modifying  ```process_dataset.py``` .
```shell script
$ process_dataset.py
```

This code automatically generates neural architecturess and then 
selects high-performing architectures among the candidates.
By setting ```--data-name``` as the name of dataset (i.e. ```cifar10```, ```cifar100```, ```aircraft100```, ```pets```), 
you can evaluate the specific dataset.

```shell script
# Meta-testing
$ python main.py --gpu 0 --model generator --hs 56 --nz 56 --test --load-epoch 120 --num-gen-arch 200 --data-name {DATASET_NAME}
```

### Arhictecture Evaluation (MetaD2A vs NSGANetV2)
##### Dataset Preparation
You need to download Oxford-IIIT Pet dataset to evaluate on ```--data-name pets```
```shell script
$ python get_files/get_pets.py
```
Every others ```cifar10```, ```cifar100```, ```aircraft100``` will be downloaded automatically.

##### evaluation
You can run the searched architecture by running ```evaluation/main```. Codes are based on NSGANetV2.

Go to the evaluation folder (i.e. ```evaluation```)
```shell script
$ cd evaluation
```

This automatically run the top 1 predicted architecture derived by MetaD2A. 
```shell script
python main.py --data-name cifar10 --num-gen-arch 200
```
You can also give flop constraint by using ```bound``` option. 
```shell script
python main.py --data-name cifar10 --num-gen-arch 200 --bound 300
```

You can compare MetaD2A with NSGANetV2 
but you need to download some files provided 
by [NSGANetV2](https://github.com/human-analysis/nsganetv2)

```shell script
python main.py --data-name cifar10 --num-gen-arch 200 --model-config flops@232
```


## Meta-Training MetaD2A Model
To build database for Meta-training, you need to set ```IMGNET_PATH```, which is a directory of ILSVRC2021.

### Database Building for Predictor
We recommend you to run the multiple ```create_database.sh``` simultaneously to build fast. 
You need to set ```IMGNET_PATH``` in the shell script.
```shell script
# Examples
bash create_database.sh 0,1,2,3 0 49 predictor
bash create_database.sh all 50 99 predictor
...
```
After enough dataset is gathered, run ```build_database.py``` to collect them as one file. 
```shell script
python build_database.py --model_name predictor --collect
```

We also provide the database we use. To download database, run ```get_files/get_predictor_database.py```: 
```shell script
$ python get_files/get_predictor_database.py
```

### Meta-Train Predictor
You can train the predictor as follows
```shell script
# Meta-training for predictor
$ python main.py --gpu 0 --model predictor --hs 512 --nz 56 
```
### Database Building for Generator
We recommend you to run the multiple ```create_database.sh``` simultaneously to build fast.
```shell script
# Examples
bash create_database.sh 4,5,6,7 0 49 generator
bash create_database.sh all 50 99 generator
...
```
After enough dataset is gathered, run ```build_database.py``` to collect them as one. 
```shell script
python build_database.py --model_name generator --collect
```

We also provide the database we use. To download database, run ```get_files/get_generator_database.py``` 
```shell script
$ python get_files/get_generator_database.py
```


### Meta-Train Generator
You can train the generator as follows
```shell script
# Meta-training for generator
$ python main.py --gpu 0 --model generator --hs 56 --nz 56 
```



## Citation
If you found the provided code useful, please cite our work.
```
@inproceedings{
    lee2021rapid,
    title={Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets},
    author={Hayeon Lee and Eunyoung Hyung and Sung Ju Hwang},
    booktitle={ICLR},
    year={2021}
}
```

## Reference
- [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks (ICML2019)](https://github.com/juho-lee/set_transformer)
- [D-VAE: A Variational Autoencoder for Directed Acyclic Graphs, Advances in Neural Information Processing Systems (NeurIPS2019)](https://github.com/muhanzhang/D-VAE)
- [Once for All: Train One Network and Specialize it for Efficient Deployment (ICLR2020)](https://github.com/mit-han-lab/once-for-all)
- [NSGANetV2: Evolutionary Multi-Objective Surrogate-Assisted Neural Architecture Search (ECCV2020)](https://github.com/human-analysis/nsganetv2)
