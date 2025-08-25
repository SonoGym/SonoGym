ARG ISAACLAB_BASE_IMAGE_ARG

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install pyvista
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install ruamel.yaml
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install pydicom
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install nibabel
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install imitation
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install monai

# # Install whatever you need as additional dependencies.
RUN bash -i -c "source ${HOME}/.bashrc && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/spinal_surgery && \
    pip install -e ."

# install fsrl
# RUN bash -i -c "cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH} && \
#     cd fsrl && \
#     pip install -e ."

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace
