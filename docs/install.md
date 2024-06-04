#### Linux
```
# check nvidia driver version >= 525.60.13
cat /proc/driver/nvidia/version

# check python version is 3.10 or higher
python --verison

pip install lacss
```

#### Windows
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
