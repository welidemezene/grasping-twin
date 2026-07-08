FROM nvcr.io/nvidia/isaac-sim:4.5.0
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
RUN /isaac-sim/python.sh -m pip install setuptools --quiet
RUN /isaac-sim/python.sh -m pip install --no-build-isolation isaaclab --quiet
RUN /isaac-sim/python.sh -m pip install stable-baselines3 --quiet
RUN /isaac-sim/python.sh -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
RUN /isaac-sim/python.sh -m pip install jupyterlab --quiet
ENTRYPOINT ["/isaac-sim/python.sh"]
