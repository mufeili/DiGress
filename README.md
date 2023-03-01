# DiGress: Discrete Denoising diffusion models for graph generation (ICLR 2023)


Warning: The code has been updated after experiments were run for the paper. If you don't manage to reproduce the
paper results, please write to us so that we can investigate the issue.

For the conditional generation experiments, check the `guidance` branch.

## Environment installation
  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit: `conda create -c conda-forge -n digress rdkit python=3.9`
  - Install graph-tool (https://graph-tool.skewed.de/): `conda install -c conda-forge graph-tool`
  - Install the nvcc drivers for your cuda version. For example, `conda install -c "nvidia/label/cuda-11.3.1" cuda-nvcc`
  - Install pytorch 1.10 or 1.11 (https://pytorch.org/)
  - Install pytorch-geometric. Your version should match the pytorch version that is installed (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  - Install other packages using the requirement file: `pip install -r requirements.txt`
  - Run `pip install -e .`
  - Navigate to the ./src/analysis/orca directory and compile orca.cpp: `g++ -O2 -std=c++11 -o orca orca.cpp`


## Download the data

  - QM9 should download by themselves when you run the code.
  - For the community dataset, data can be found at https://github.com/KarolisMart/SPECTRE/tree/main/data



## Run the code

  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - You can specify the dataset with `python3 main.py dataset=comm20`. Look at `configs/dataset` for the list of datasets that are currently available

## Generated samples

We provide the generated samples for some of the models. If you have retrained a model from scratch for which the samples are
not available yet, we would be very happy if you could send them to us!

## Cite the paper

```
@inproceedings{
vignac2023digress,
title={DiGress: Discrete Denoising diffusion for graph generation},
author={Clement Vignac and Igor Krawczuk and Antoine Siraudin and Bohan Wang and Volkan Cevher and Pascal Frossard},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=UaAD-Nu86WX}
}
```
