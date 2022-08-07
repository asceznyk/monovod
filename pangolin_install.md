`
git clone https://github.com/uoip/pangolin.git
cd pangolin
mkdir build
cd build
cmake -DBUILD_PANGOLIN_FFMPEG=OFF -DPYBIND11_PYTHON_VERSION=3.8 ..
make -j8
cd ..
python3 setup.py install
`
