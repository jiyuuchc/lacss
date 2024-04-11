#### Linux - GPU
```
# check nvidia driver version >= 450.80.02
cat /proc/driver/nvidia/version

# check python version is 3.10 or higher
python --verison

# install jax and lacss
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install lacss
```

#### Windows - GPU
  1. Make sure nividia driver version >= 452.39, following [Nvidia-FAQ](https://www.nvidia.com/en-gb/drivers/drivers-faq/)
  2. Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
  3. In a miniconda shell/cmd:
```
conda create -n lacss python=3.10
conda activate lacss

conda install cudnn=8.9.2.26
conda install cuda-nvcc -c nvidia
conda install zlib-wapi -c conda-forge
pip install https://whls.blob.core.windows.net/unstable/cuda118/jaxlib-0.4.11+cuda11.cudnn86-cp310-cp310-win_amd64.whl
pip install jax==0.4.11 ml-dtypes==0.2 flax==0.7.2
pip install lacss
```

#### Linux / Windows - CPU
```
pip install lacss
```
