FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
# FROM ubuntu:16.04
MAINTAINER Li Shen <shenli.sam@gmail.com>
# FROM nvidia/cuda:8.0-devel
# LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# KEEP UBUNTU OR DEBIAN UP TO DATE
RUN echo "deb http://us.archive.ubuntu.com/ubuntu xenial main multiverse" >> /etc/apt/sources.list
RUN apt-get -y update 
RUN apt-get -y install apt-utils
ENV TERM xterm-256color  # just to turn off some warnings.
RUN apt-get -y upgrade \
    && apt-get -y autoremove
RUN apt-get install -y wget unzip

## ================= CUDA 8 ===================== #
#WORKDIR /
#RUN wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
#RUN apt-get -y update && apt-get install -y curl perl
#RUN apt-get install -y make
#RUN curl -L https://cpanmin.us | perl - App::cpanminus
#RUN cpanm Tie::File
#RUN bash ./cuda_8.0.44_linux-run --silent
#ENV PATH /usr/local/cuda-8.0/bin${PATH:+:${PATH}}
#ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#
## COPY cuda-repo-ubuntu1604_8.0.44-1_amd64.deb ./
## RUN dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
## RUN apt-get -y update && apt-get install -y cuda
#
## ================= CUDNN 5 =================== #
#RUN rm -rf /var/lib/apt/lists/*
#
#ENV CUDNN_VERSION 5
#LABEL com.nvidia.cudnn.version="5"
#
#RUN CUDNN_DOWNLOAD_SUM=a87cb2df2e5e7cc0a05e266734e679ee1a2fadad6f06af82a76ed81a23b102c8 && \
#    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz -O && \
#    echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v5.1.tgz" | sha256sum -c --strict - && \
#    tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local && \
#    rm cudnn-8.0-linux-x64-v5.1.tgz && \
#    ldconfig
    
# =================== OpenCV ====================== #
# Build tools.
RUN apt-get -y update && apt-get install -y build-essential cmake git

# OpenCV2 development, parallelism, linear algebra.
RUN apt-get install -y libtbb-dev libeigen3-dev
# libopencv-dev 

# 1394 Camera.
RUN apt-get install -y libdc1394-22 libdc1394-22-dev

# GUI (if you want to use GTK instead of Qt, replace 'qt5-default' with 'libgtkglext1-dev' and remove '-DWITH_QT=ON' option in CMake):
RUN apt-get -y update && apt-get install -y qt5-default libgdal-dev libvtk6-dev libgtk-3-dev qtbase5-dev

# Media I/O:
RUN apt-get install -y zlib1g-dev libwebp-dev libjasper-dev libopenexr-dev libavcodec-dev libavformat-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libvorbis-dev

# Video I/O:
RUN apt-get install -y libx264-dev yasm x264 ffmpeg libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev v4l-utils libtheora-dev libxvidcore-dev

# Python:
RUN apt-get install -y python-dev python-tk python-numpy python-scipy python3-dev python3-tk python3-numpy python3-scipy

# Documentation:
RUN apt-get install -y doxygen

# Additional packages (Later added when tested on my Ubuntu 16.04 box):
RUN apt-get install -y libavresample-dev libgphoto2-dev
RUN apt-get install -y python-gst-1.0 python-gst0.10 python-gst0.10-dev python3-gst-1.0

# Image I/O:
RUN apt-get install -y libjpeg-dev libpng12-dev libswscale-dev

# INSTALL THE OpenCV LIBRARY (YOU CAN CHANGE '3.1.0' FOR THE LAST STABLE VERSION)
WORKDIR /
RUN git clone https://github.com/Itseez/opencv.git
WORKDIR opencv
RUN git config --global user.email "shenli.sam@gmail.com"
RUN git config --global user.name "Li Shen"
RUN git checkout 3.1.0 && \
    git format-patch -1 10896129b39655e19e4e7c529153cb5c2191a1db && \
    git am < 0001-GraphCut-deprecated-in-CUDA-7.5-and-removed-in-8.0.patch
# RUN wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O 3.1.0.zip
# RUN unzip 3.1.0.zip \
#     && rm 3.1.0.zip \
#     && mv opencv-3.1.0 OpenCV
# WORKDIR opencv
WORKDIR build
RUN cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_GSTREAMER=OFF \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES;-Wno-deprecated-gpu-targets" \
    -D WITH_TIFF=ON -D BUILD_TIFF=ON \
    -D WITH_OPENMP=ON \
    -D BUILD_EXAMPLES=OFF \
    ..

# make and install.
RUN make clean && make -j7 && make install
RUN bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && ldconfig

# >>>>> Also need to create a link to cv2.so if anaconda is being used. <<<<<<<
# RUN ln -s /usr/local/lib/python2.7/site-packages/cv2.so /anaconda/lib/python2.7/cv2.so
RUN python -c "import cv2; print cv2.__version__"

# ============================================== #
RUN apt-get -y update && apt-get install -y parallel imagemagick

# ===================== Tensorflow ======================#
ENV PATH /usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-367-dev
RUN apt-get install -y python-pip python-dev && pip install -U pip 
RUN pip install -U tensorflow-gpu

# ====================== Sklearn ========================#
RUN pip install -U scikit-learn
RUN python -c "import sklearn; print sklearn.__version__"

# ====================== Keras ==========================#
RUN pip install -U pyyaml six h5py pydot-ng
RUN git clone https://github.com/lishen/keras.git
WORKDIR keras
RUN git checkout fix-1-channel-samplewise-std
RUN python setup.py install
RUN python -c "import keras; print keras.__version__"
# WORKDIR /
# RUN pip install ./keras

# ============================================== #
# Copy processing and training files:
WORKDIR /
COPY train.sh ./
COPY temp/*.py dm_resnet_train.py ./



