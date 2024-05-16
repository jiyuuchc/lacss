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
  - Make sure nividia driver version >= 452.39, following [Nvidia-FAQ](https://www.nvidia.com/en-gb/drivers/drivers-faq/)
  - Make sure you have a python version >= 3.10 (python --version)
  - Install CUDA and cudnn following Nivida's step-by-step instructions [Nvidai-cudnn](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html)
  - Download jaxlib wheel from [jaxlib](https://whls.blob.core.windows.net/unstable/index.html). The file name specifies versions of cuda, cudnn and python. Make sure it matches your setup.
  - Install lacss and dependencies into your virtual environment
  
```
pip install <downloaded jaxlib wheel file>
pip install jax==<jaxlib version> flax==0.7.2
pip install lacss
```

#### Linux / Windows - CPU
```
pip install lacss
```
