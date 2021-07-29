# Compare IPM Base GAN
This is compare base Gan Repository

## How to use
Run the script using: <br>
```CUDA_VISIBLE_DEVICES= <gpu_usages> python main.py -c <config_path> <--train| --eval>``` <br>
example : <br>
```CUDA_VISIBLE_DEVICES= 0,1 python main.py -c configs/mnist_mlp_wgan_gp.yaml --train``` <br>

## Configs :
There are two section for each config, separated by `---`
#### 1. Global config (first)
Put all global variable here.
#### 2. Modules (second)
Put all configuration for each module here, you can use jinja method on this section from global config `{{ variable }}. <br>
A module config, example:
```yaml
name : module_a  # only name
module : ModuleA # class module name 
configs: # all configuration of module
    a: pass
    b: pass
```
If there are multiple modules add 'modules' on configs, example:
```yaml
modules:
    - name : first_module # only name
      module : FirstModule # class module name 
      configs: # all configuration of module
            a: pass
            b: pass

    - name : second_module # only name
      module : SecondModule # class module name 
      configs: # all configuration of module
            a: pass
            b: pass
```
You also can set moduleclass using `.`, example

```yaml
modules:
    - name : critic # naming as critic module
      module : sinkhron.Critic # get Critic Class from sinkhorn file 
      configs: # all configuration of module
            a: pass
            b: pass

    - name : generator # naming as generator module
      module : sinkhorn.Generator # get Generator Class from sinkhorn file  
      configs: # all configuration of module
            a: pass
            b: pass
```
#### Example:
```yaml
# Global config and can be use as namespace
root_dir: ../experiments
dataset: Mnist
trainer: wgan
model: mlp
optim: RMSprop
gpu: true
n_gpu_use: 1
ngf: 64
ndf: 64
# Data input
batch_size:  100
input_size:  [1, 28, 28]
z_dim:  100
critic_dim: 100
n_epoch: 100000
note: GP
# will Create root_dir/dataset/trainer/model/note

---
# Session Modules
DataLoader:
  modules:
    - name: main
      module: {{ dataset }}DataLoader
      configs:
        size: {{ input_size }}
        batch_size: {{ batch_size }}
        shuffle: true
        sampler: ~
        batch_sampler: ~
        num_workers: 2  # default 4
        collate_fn: ~
        pin_memory: false
        drop_last: false
        timeout: 0
        worker_init_fn: ~
Model:
  modules:
    - name: critic
      network:
        module: {{ model }}.Critic
        configs:
          ndf: {{ ndf }}
          z_dim: {{ z_dim }}
          input_size: {{ input_size }}
          critic_dim: {{ critic_dim }}
      optim:
        module: {{ optim }}
        configs:
          lr: 0.0001
    - name: generator
      network:
        module: {{ model }}.Generator
        configs:
          ngf: {{ ngf }}
          z_dim: {{ z_dim }}
          input_size: {{ input_size }}
          critic_dim: {{ critic_dim }}
      optim:
        module: {{ optim }}
        configs:
          lr: 0.0001
Logger:
  modules:
    - name: scalar
      module: Scalar
Evaluator:
  modules:
    - name: generated_image
      module: GeneratedImage
      configs:
        run_per: [1000, generator]
        size: [64, {{ z_dim }}]
    # - name: is
    #   module: IS
    #   configs:
    #     run_per: [1, epoch]
    #     n_iter: 1000
    #     batch_size: 32
    #     size: [64, {{ z_dim }}]
    # - name: fid
    #   module: FID
    #   configs:
    #     run_per: [1, epoch]
    #     n_iter: 1000
    #     batch_size: 32
    #     size: [64, {{ z_dim }}]
Trainer:
  configs:
    save_model_per: [20, epoch] # can select epoch or step
    save_log_per: [200, step] # can select epoch or step
  updater:
    module: GAN
    configs:
      z_dim: {{ z_dim }}
      fixed: false
  modules:
    - name: critic
      criterion:
        module: {{ trainer }}.Critic
      regularization:
        modules:
          - name: gp
            module: GradientPenalty 
            configs:
              lambda: 0.4

    - name: generator
      criterion:
        module: {{ trainer }}.Generator
```
## Repo Structure:
The repo has the following structure:
```
├── base # all base for many purposes
│   ├── container.py
│   ├── criterion.py
│   ├── dataloader.py
│   ├── evaluator.py
│   ├── network.py
│   ├── properties.py
│   └── updater.py
├── configs # put your config file here
│   ├── config_template.yaml
│   ├── mnist_mlp_sinkhorn_gan.yaml
│   ├── mnist_mlp_sinkhorn_sgd.yaml
│   ├── mnist_mlp_wgan_div.yaml
│   └── mnist_mlp_wgan_gp.yaml
├── constructor # constructor Module
│   ├── dataloader.py
│   ├── evaluator.py
│   ├── logger.py
│   ├── model.py
│   ├── regularization.py
│   ├── session.py
│   └── trainer.py
├── main.py
├── module # you only edit here if want adding new method
│   ├── dataloader # dataloader list
│   │   ├── cat.py
│   │   ├── celeba.py
│   │   ├── cifar.py
│   │   ├── datasets_dict.py
│   │   ├── Datasets.py
│   │   ├── dogvscat.py
│   │   ├── mnist.py
│   ├── evaluator # evaluator model
│   │   ├── fid.py
│   │   ├── generated_image.py
│   │   ├── inception_score.py
│   ├── logger # logging
│   │   ├── logger.yaml
│   │   └── scalar.py
│   ├── model # all model put here
│   │   ├── network
│   │   │   ├── dcgan.py
│   │   │   ├── densenet_noise.py
│   │   │   ├── densenet.py
│   │   │   ├── encoder_decoder.py
│   │   │   ├── mlp.py
│   │   │   └── resnet.py
│   │   └── regularization
│   │       ├── spectral_normalization.py
│   │       └── weight_clipping.py
│   └── trainer
│       ├── criterion
│       │   ├── mmd.py
│       │   ├── ragan.py
│       │   ├── rawgan.py
│       │   ├── sinkhorn_gan.py
│       │   ├── sinkhorn_sgd.py
│       │   └── wgan.py
│       ├── post
│       ├── pre
│       ├── regularization
│       │   ├── gradient_penalty.py
│       │   └── wasserstain_div.py
│       └── updater
│           ├── gan.py
├── README.md
├── requirements.txt
└── utils
    ├── __init__.py
    └── utils.py

```
