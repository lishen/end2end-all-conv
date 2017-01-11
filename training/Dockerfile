FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Li Shen <shenli.sam@gmail.com>

# ============== OpenCV2, numpy, scipy =================== #
RUN echo "deb http://us.archive.ubuntu.com/ubuntu xenial main multiverse" >> /etc/apt/sources.list
ENV TERM xterm-256color  # just to turn off some warnings.
RUN apt-get -y update && \
	apt-get install -y --no-install-recommends python-opencv python-numpy python-scipy && \
	apt-get autoclean && \
	apt-get autoremove
RUN python -c "import cv2; print cv2.__version__"

# =======================Parallel & convert ==================== #
RUN apt-get -y update && \
	apt-get install -y --no-install-recommends parallel imagemagick && \
	apt-get autoclean && \
	apt-get autoremove	

# ===================== Tensorflow ======================#
ENV PATH /usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nvidia-367-dev && \
	apt-get autoclean && \
	apt-get autoremove
RUN apt-get install -y --no-install-recommends python-pip python-dev && \
	pip install -U pip && \
	apt-get autoclean && \
	apt-get autoremove	
RUN pip install -U tensorflow-gpu

# ====================== Sklearn ========================#
RUN pip install -U scikit-learn
RUN python -c "import sklearn; print sklearn.__version__"

# ====================== Keras ==========================#
RUN pip install -U pyyaml six h5py pydot-ng
WORKDIR /
RUN echo "Keras last updated: 2016-12-22"
RUN apt-get install -y --no-install-recommends git && \
	git clone https://github.com/lishen/keras.git
WORKDIR keras
RUN git checkout fix-1-channel-samplewise-std
RUN python setup.py install
RUN python -c "import keras; print keras.__version__"

# ============================================== #
# Copy processing and training files:
RUN pip install -U pandas pydicom
WORKDIR /
COPY train.sh train_small.sh ./
COPY temp/*.py dm_resnet_train.py ./
COPY modelState/2017-01-11_resnet47rb5_576/dmresnet47rb5_576_bestAuc_model.h5 ./
# VOLUME ["/metadata", "/trainingData", "/preprocessedData", "/modelState", "/scratch"]



