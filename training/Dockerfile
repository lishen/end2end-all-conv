FROM ubuntu:16.04
MAINTAINER Li Shen <shenli.sam@gmail.com>

# KEEP UBUNTU OR DEBIAN UP TO DATE
RUN echo "deb http://us.archive.ubuntu.com/ubuntu xenial main multiverse" >> /etc/apt/sources.list
RUN apt-get -y update 
RUN apt-get -y install apt-utils
ENV TERM xterm-256color  # just to turn off some warnings.
RUN apt-get -y upgrade \
    && apt-get -y autoremove

    
# =================== OpenCV ====================== #
# Build tools.
RUN apt-get install -y build-essential cmake git

# OpenCV2 development, parallelism, linear algebra.
RUN apt-get install -y libtbb-dev libeigen3-dev
# libopencv-dev 

# 1394 Camera.
RUN apt-get install -y libdc1394-22 libdc1394-22-dev

# GUI (if you want to use GTK instead of Qt, replace 'qt5-default' with 'libgtkglext1-dev' and remove '-DWITH_QT=ON' option in CMake):
RUN apt-get install -y qt5-default libvtk6-dev libgtk-3-dev qtbase5-dev

# Media I/O:
RUN apt-get install -y zlib1g-dev libwebp-dev libjasper-dev libopenexr-dev libgdal-dev libavcodec-dev libavformat-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libvorbis-dev

# Video I/O:
RUN apt-get install -y libx264-dev yasm x264 ffmpeg libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev v4l-utils libtheora-dev libxvidcore-dev

# Anaconda:
# WORKDIR /
# RUN wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh -O anaconda.sh
# RUN bash anaconda.sh -b -p anaconda
# ENV PATH "/anaconda/bin:$PATH"
# RUN conda upgrade --all -y
# RUN echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Python:
RUN apt-get install -y python-dev python-tk python-numpy python3-dev python3-tk python3-numpy

# Documentation:
RUN apt-get install -y doxygen

# Additional packages (Later added when tested on my Ubuntu 16.04 box):
RUN apt-get install -y libavresample-dev libgphoto2-dev
RUN apt-get install -y python-gst-1.0 python-gst0.10 python-gst0.10-dev python3-gst-1.0

# Image I/O:
RUN apt-get install -y libjpeg-dev libpng12-dev libswscale-dev
# libtiff5 libtiff5-dev

# INSTALL THE OpenCV LIBRARY (YOU CAN CHANGE '3.1.0' FOR THE LAST STABLE VERSION)
WORKDIR /
RUN apt-get install -y wget unzip
RUN wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O 3.1.0.zip
RUN unzip 3.1.0.zip \
    && rm 3.1.0.zip \
    && mv opencv-3.1.0 OpenCV
WORKDIR OpenCV
WORKDIR build
RUN cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_GSTREAMER=OFF \
    -D WITH_TIFF=ON -D BUILD_TIFF=ON \
    -D WITH_OPENMP=ON \
    -D BUILD_EXAMPLES=OFF \
    -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" \
    # -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    # -D PYTHON2_EXECUTABLE=/anaconda/bin/python \
    # -D PYTHON_INCLUDE_DIR=/anaconda/include/python2.7/ \
    # -D PYTHON_LIBRARY=/anaconda/lib/libpython2.7.so \
    # -D PYTHON2_NUMPY_INCLUDE_DIRS=/anaconda/lib/python2.7/site-packages/numpy/core/include/ \
    ..

# make and install.
RUN make clean && make -j7 && make install
RUN bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && ldconfig

# >>>>> Also need to create a link to cv2.so if anaconda is being used. <<<<<<<
# RUN ln -s /usr/local/lib/python2.7/site-packages/cv2.so /anaconda/lib/python2.7/cv2.so
RUN python -c "import cv2; print cv2.__version__"


# ============================================== #
RUN apt-get install -y parallel imagemagick


# ============================================== #
# Copy processing and training files:
WORKDIR /
COPY preprocess.py ./
COPY preprocess.sh ./
COPY train.sh ./





