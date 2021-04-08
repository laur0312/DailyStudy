### Docker

参考网址

https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2019

https://ngc.nvidia.com/catalog/containers/nvidia:pytorch



1. 运行docker

   ```
   sudo docker run --rm -it nvcr.io/nvidia/pytorch:20.08-py3 bash
   ```

2. 编译nvidia-docker

   ```
   sudo nvidia-docker build -t alpha-devel:3.0 -f Dockerfile_3.0 .
   ```
   
3. 序列化+压缩存储

   ```
   docker save alpha-devel:3.0 | gzip >  alpha-devel_3.0.tar.gz
   ```

4. 加载

   ```
   gunzip -c alpha-devel_3.0.tar.gz | docker load
   ```

5. 上传harbour

   ```
   1. login harbour
   docker login http://harbor.do.proxima-ai.com 
   harbor用户名 | 密码：alpha | Alpha987654
   2. 给镜像打tag
   docker tag alpha-devel:3.0 harbor.do.proxima-ai.com/alpha/alpha-devel:3.0
   3. push to harbour 
   docker push harbor.do.proxima-ai.com/alpha/alpha-devel:3.0
   ```




踩坑记录

- 通过docker-compose -f ./docker-compose.yml启动时，environment的环境变量并未生效，此时需要在dockerfile中将环境变量写入/etc/profile
- dockerfile中entrypoint和CMD的差异
- tail -f /dev/null可使docker非立即退出



Dockerfile_3.0文件

```dockerfile
FROM nvcr.io/nvidia/pytorch:20.08-py3
MAINTAINER yanghua@fosun.com
LABEL version="3.0"

ENV PATH=/usr/local/bin:$PATH\
    LANG=C.UTF-8\
    WORK_PATH=/home\
    _MAKE_FILE_PATH=file\
    DEBIAN_FRONTEND=noninteractive

WORKDIR $WORK_PATH
COPY $_MAKE_FILE_PATH/code_*.deb $WORK_PATH/code_*.deb
COPY $_MAKE_FILE_PATH/nccl-repo-*.deb $WORK_PATH/nccl-repo-*.deb
COPY $_MAKE_FILE_PATH/apex.tar.gz $WORK_PATH/apex.tar.gz

# python-lib
RUN pip3 install --no-cache-dir SimpleITK==1.1.0 \
                                sympy==1.1.1 \
                                nose==1.3.7 \
                                Crypto==1.4.1 \
                                pycrypt==0.7.2 \
                                pycrypto==2.6.1 \
                                vtk==8.1.1 \
                                stomp.py==4.1.21 \
                                pydicom==1.2.0 \
                                easydict==1.9 \
                                rpy2==3.0.3 \
                                pywavelets==1.0.0 \
                                pyradiomics==2.1.2 \
                                trimesh==2.38.8 \
                                Shapely==1.6.4.post1 \
                                ipython \
                                ITK==4.13.2 \
                                keras==2.2.4 \
                                yacs \
                                openpyxl \
                                xlrd \
                                lmdb \
                                imgaug \
                                pynrrd \
                                tensorboardX \
                                wget \
                                progressbar2 \
                                nibabel

# remove old nccl2
Run apt-get remove libnccl2 libnccl-dev -y 

# install ssh, gdb, sudo, vscode, python3-tk, screen, tmux, system-monitor, nccl2
RUN dpkg -i nccl-repo-*.deb
RUN apt-get update \
    && apt-get install ./code_*.deb -y \
    && apt-get install libasound2 -y \
    && apt-get install sudo -y \
    && apt-get install openssh-server -y \
    && apt-get install gdb gdbserver -y \
    && apt-get install xorg -y \
    && apt-get install python3-tk -y \
    && apt-get install screen -y \
    && apt-get install tmux  -y \
    && apt-get install net-tools -y \
    && apt-get install dbus-x11 -y \
    && apt-get install gnome-system-monitor -y \
    && apt-get install libnccl2 -y \
    && apt-get install libnccl-dev -y \
    && apt-get clean

# install openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
RUN tar -xvf openmpi-4.0.1.tar.gz
WORKDIR $WORK_PATH/openmpi-4.0.1
RUN ./configure --prefix=/usr/local && make -j 8 all && make install
WORKDIR $WORK_PATH

# install horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --no-cache-dir horovod

# install Nvidia Apex
RUN tar -xvf apex.tar.gz
WORKDIR $WORK_PATH/apex
RUN pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
WORKDIR $WORK_PATH

# config ssh
RUN sed -i '$a PermitRootLogin yes' /etc/ssh/sshd_config \
    && sed -i '$a X11UseLocalhost no' /etc/ssh/sshd_config \
    && echo "root:123456" | chpasswd

RUN echo "Asia/Shanghai" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata

# set proxy
RUN sed -i '$a export http_proxy=http://172.16.17.164:3128' /etc/profile \
    && sed -i '$a export https_proxy=http://172.16.17.164:3128' /etc/profile

# add chinese
RUN sed -i '$a export LANG="C.UTF-8"' /etc/profile

# add conda
RUN sed -i '$a export PATH=/opt/conda/bin:$PATH' /etc/profile

# clean
RUN rm -rf $WORK_PATH/*

# deploy
COPY $_MAKE_FILE_PATH/startup.sh /usr/local/bin/startup.sh

# run
ENTRYPOINT ["sh"]
CMD ["/usr/local/bin/startup.sh"]
```



startup.sh

```shell
#!/bin/bash
echo begin 

# restart ssh for log
service ssh restart

#run something 
tail -f /dev/null

echo end 
```

