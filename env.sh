# note: launch this script via ". ./env.sh"

bash ./preprocess_sfm/install_colmapforvissat.sh

conda create -y -n satellite_sfm python=3.8 && conda activate satellite_sfm
pip install numpy matplotlib opencv-python pyexr open3d tqdm icecream imageio-ffmpeg
pip install utm pyproj pymap3d
conda install -y -c conda-forge gdal
pip install trimesh pyquaternion


