FROM ros:noetic-ros-core

SHELL ["/bin/bash", "-c"] 

RUN sudo apt-get update \
    && sudo apt-get install -y \
    git python3-pip python3-tf2-ros ros-noetic-foxglove-msgs \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install matplotlib notebook numpy nuscenes-devkit opencv-python-headless seaborn
RUN pip3 install git+https://github.com/DanielPollithy/pypcd.git

# RUN source /usr/local/bin/virtualenvwrapper.sh
RUN mkdir /notebooks

WORKDIR /notebooks
EXPOSE 8888/tcp

# RUN groupadd xli4217
# RUN useradd -r -u 1001 -g xli4217 xli4217
# RUN usermod -aG sudo xli4217
# USER xli4217:xli4217 

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
#CMD ["source", "/opt/ros/neotic/setup.bash"]