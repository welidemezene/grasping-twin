FROM nvcr.io/nvidia/isaac-sim:4.5.0
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
RUN /isaac-sim/python.sh -m pip install setuptools --quiet &&     /isaac-sim/python.sh -m pip install --no-build-isolation isaaclab --quiet &&     /isaac-sim/python.sh -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
ENTRYPOINT ["/isaac-sim/python.sh"]
