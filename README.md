# Convergence of Non-Convex Non-Concave GANs Using Sinkhorn Divergence

This repository is the official implementation of [Convergence of Non-Convex Non-Concave GANs Using Sinkhorn Divergence](https://ieeexplore.ieee.org/abstract/document/9410544). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
conda install --file requirements_conda.txt
pip install -r requirements_pip.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

There are two options in running experiment using our code:

1. Execute the bash script to run preset configs used in the paper:

>NOTE : uncomment any of one line from Line 4-8 to run one of configs used in our paper

```train
bash ./runs.sh
```

2. Run the experiment thru `main.py`

```
python main.py \\
	--c path_to_config_file.yaml
	--train
```

To watch the experiment, we use Tensorboard watching the experiment directory

```
tensorboard --logdir ../experiments/runs
	
```

> We suggest adding `--samples_per_plugin "scalar=0"` for more precise recording of the experiment

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
