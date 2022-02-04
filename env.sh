# note: launch this script via ". ./env.sh"

bash ./preprocess_sfm/install_colmapforvissat.sh

conda create -y -n SatelliteSfM python=3.8 && conda activate SatelliteSfM
pip install numpy matplotlib opencv-python pyexr open3d tqdm icecream imageio imageio-ffmpeg
pip install utm pyproj pymap3d
conda install -y -c conda-forge gdal
pip install trimesh pyquaternion


conda install -y -c anaconda libtiff
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
pip install srtm4



