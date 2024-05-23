conda create -n buddi python=3.10 -y
conda activate buddi
# Install pytorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# Install fvcore
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y

# Set to correct cuda path in case multiple nvcc versions are installed
export CUDA_HOME="/usr/local/cuda-11.7"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-11.7/bin:$PATH"

# Install pytorch3d (about 20-25mins)- It requires cuda version detected by default to be the same as the one used to compile pytorch!
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

## Install third party libraries
pip install opencv-python smplx scipy scikit-image loguru omegaconf ipdb einops chumpy trimesh setuptools==58.2.0
pip install tensorboard tensorboardX
pip install matplotlib numpy==1.23.1
pip install scikit-learn
pip install wandb
