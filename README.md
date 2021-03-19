# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets
This is the official **PyTorch implementation** for the paper Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets (**ICLR 2021**) : https://openreview.net/forum?id=rkQuFUmUOg3.

## Abstract
<img align="middle" width="700" src="https://github.com/HayeonLee/tmp/blob/main/images/MetaD2A_concept.png">

Despite the success of recent Neural Architecture Search (NAS) methods on various tasks which have shown to output networks that largely outperform human-designed networks, conventional NAS methods have mostly tackled the optimization of searching for the network architecture for a single task (dataset), which does not generalize well across multiple tasks (datasets). Moreover, since such task-specific methods search for a neural architecture from scratch for every given task, they incur a large computational cost, which is problematic when the time and monetary budget are limited. In this paper, we propose an efficient NAS framework that is trained once on a database consisting of datasets and pretrained networks and can rapidly search a neural architecture for a novel dataset. The proposed MetaD2A (Meta Dataset-to-Architecture) model can stochastically generate graphs (architectures) from a given set (dataset) via a cross-modal latent space learned with amortized meta-learning. Moreover, we also propose a meta-performance predictor to estimate and select the best architecture without direct training on target datasets. The experimental results demonstrate that our model meta-learned on subsets of ImageNet-1K and architectures from NAS-Bench 201 search space successfully generalizes to multiple benchmark datasets including CIFAR-10 and CIFAR-100, with an average search time of 33 GPU seconds. Even under a large search space, MetaD2A is 5.5K times faster than NSGANetV2, a transferable NAS method, with comparable performance. We believe that the MetaD2A proposes a new research direction for rapid NAS as well as ways to utilize the knowledge from rich databases of datasets and architectures accumulated over the past years. 

__Framework of MetaD2A Model__

<img align="middle" width="700" src="https://github.com/HayeonLee/tmp/blob/main/images/MetaD2A_model.png">

## Prerequisites
- Python 3.6 (Anaconda)
- PyTorch 1.6.0
- CUDA 10.2
- python-igraph==0.8.2
- tqdm==4.50.2
- torchvision==0.7.0
- python-igraph==0.8.2
- nas-bench-201==1.3
- scipy==1.5.2


If you are not familiar with preparing conda environment, please follow the below instructions
```
$ conda create --name metad2a python=3.6
$ conda activate metad2a
$ conda install pytorch==1.6.0 torchvision cudatoolkit=10.2 -c pytorch
$ pip install nas-bench-201
$ conda install -c conda-forge tqdm
$ conda install -c conda-forge python-igraph
$ pip install scipy
```

And for data preprocessing,
```
$ pip install requests
```

Hardware Spec used for experiments of the paper
- GPU: A single Nvidia GeForce RTX 2080Ti
- CPU: Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz

## NAS-Bench-201
Go to the folder for NAS-Bench-201 experiments (i.e. ```MetaD2A_nas_bench_201```)
```
$ cd MetaD2A_nas_bench_201
```

### Data Preparation
To download preprocessed data files, run ```get_files/get_preprocessed_data.py```: 
```
$ python get_files/get_preprocessed_data.py
```
It will take some time to download and preprocess each dataset.

To download MNIST, Pets and Aircraft Datasets, run ```get_files/get_{DATASET}.py```
```
$ python get_files/get_mnist.py
$ python get_files/get_aircraft.py
$ python get_files/get_pets.py
```

Other datasets such as Cifar10, Cifar100, SVHN will be automatically downloaded when you load dataloader by torchvision.

### MetaD2A Evaluation (Meta-Test)

You can download trained checkpoint files for generator and predictor
```
$ python get_files/get_checkpoint.py
$ python get_files/get_predictor_checkpoint.py
```

#### 1. Evaluation on __Cifar10 and Cifar100__

By set ```--data-name``` as the name of dataset (i.e. ```cifar10```, ```cifar100```), 
you can evaluate the specific dataset only
```
# Meta-testing for generator 
$ python main.py --gpu 0 --model generator --hs 56 --nz 56 --test --load-epoch 400 --num-gen-arch 500 --data-name {DATASET_NAME}
```

After neural architecture generation is completed,
meta-performance predictor selects high-performing architectures among the candidates
```
# Meta-testing for predictor
$ python main.py --gpu 0 --model predictor --hs 512 --nz 56 --test --num-gen-arch 500 --data-name {DATASET_NAME}
```

#### 2. Evaluation on __Other Datasets__

By set ```--data-name``` as the name of dataset (i.e. ```mnist```, ```svhn```, ```aircraft```, ```pets```), 
you can evaluate the specific dataset only
```
# Meta-testing for generator
$ python main.py --gpu 0 --model generator --hs 56 --nz 56 --test --load-epoch 400 --num-gen-arch 50 --data-name {DATASET_NAME}
```

After neural architecture generation is completed,
meta-performance predictor selects high-performing architectures among the candidates
```              
# Meta-testing for predictor
$ python main.py --gpu 0 --model predictor --hs 512 --nz 56 --test --num-gen-arch 50 --data-name {DATASET_NAME}
```


### Meta-Training MetaD2A Model
You can train the generator and predictor as follows
```
# Meta-training for generator
$ python main.py --gpu 0 --model generator --hs 56 --nz 56 
                 
# Meta-training for predictor
$ python main.py --gpu 0 --model predictor --hs 512 --nz 56 
```

### Results
The results of training architectures which are searched by meta-trained MetaD2A model for each dataset

__Accuracy__
|                         |     CIFAR10    |    CIFAR100    |      MNIST     |      SVHN      |    Aircraft    | Oxford-IIT Pets |
|:-----------------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:---------------:|
|         PC-DARTS        |   93.66±0.17   |   66.64±0.04   |   99.66±0.04   |   95.40±0.67   |   46.08±7.00   |    25.31±1.38   |
| __MetaD2A__  __(Ours)__ | __94.37±0.03__ | __73.51±0.00__ | __99.71±0.08__ | __96.34±0.37__ | __58.43±1.18__ |  __41.50±4.39__ |

__Search Time (GPU Sec)__
|                         |  CIFAR10  |  CIFAR100 |   MNIST   |    SVHN   |  Aircraft | Oxford-IIT Pets |
|:-----------------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------------:|
|         PC-DARTS        |   10395   |   19951   |   24857   |   31124   |    3524   |       2844      |
| __MetaD2A__  __(Ours)__ | __69__ | __96__ | __7__ | __7__ | __10__ |    __8__    |



## MobileNetV3 Search Space
Go to the folder for MobileNetV3 Search Space experiments (i.e. ```MetaD2A_ofa```)
```
$ cd MetaD2A_ofa
```
And follow [README.md](https://github.com/HayeonLee/MetaD2A/blob/main/MetaD2A_ofa/README.md) written for experiments of __OFA Search Space__

## Your Own Dataset
- [ ] It will be updated!

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
- [NAS-BENCH-201: Extending the Scope of Reproducible Neural Architecture Search (ICLR2020)](https://github.com/D-X-Y/NAS-Bench-201)
- [Once for All: Train One Network and Specialize it for Efficient Deployment (ICLR2020)](https://github.com/mit-han-lab/once-for-all)
