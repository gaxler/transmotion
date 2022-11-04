rm -rf /content/transmotion/ 
git clone https://github.com/gaxler/transmotion.git
pip install -r /content/transmotion/requirements.txt
pip install einops
pip install --upgrade imageio[ffmpeg]
wget -c https://cloud.tsinghua.edu.cn/f/da8d61d012014b12a9e4/?dl=1 -O /content/transmotion/vox.pth.tar
