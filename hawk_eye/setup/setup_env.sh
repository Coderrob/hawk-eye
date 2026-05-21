#!/usr/bin/env bash

set -e

if [ -z "$1" ]
  then
    echo "No python virtual environment specified."
    USE_VENV=0
else
    echo "Python virtual environment will be created."
    USE_VENV=1
    VENV_PATH="$1"
fi

sudo apt update && \
    sudo apt upgrade -y && \
    sudo apt install -y curl gzip python3-dev build-essential

pushd $(mktemp -d)
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
rm get-pip.py
popd
python3 -m pip install --upgrade pip

if [ $USE_VENV -eq 1 ]; then
    # Create a virtual environment for hawk_eye
    python3 -m venv $VENV_PATH
    source "$VENV_PATH/bin/activate"
fi

python3 -m pip install -U -r hawk_eye/setup/requirements.txt
pre-commit && pre-commit install

IS_MAC=$(uname -a)

if [[ $IS_MAC =~ "Darwin" ]]; then
    echo "No GPU found."
    python3 -m pip install -U torch torchvision
else
    if lspci -vnnn | perl -lne 'print if /^\d+\:.+(\[\S+\:\S+\])/' | grep -q NVIDIA;
    then
        echo "GPU found."
        python3 -m pip install -U -r hawk_eye/setup/requirements-gpu.txt
    else
        python3 -m pip install -U -r hawk_eye/setup/requirements-cpu.txt
    fi
fi

# Get bazelisk
pushd $(mktemp -d)
if [[ $IS_MAC =~ "Darwin" ]]; then
    echo "Downloading Bazelisk for Darwin"
    curl -fL -o bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.7.2/bazelisk-darwin-amd64
else
    curl -fL -o bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.7.1/bazelisk-linux-amd64
fi

chmod +x bazel
sudo mv bazel /usr/local/bin
popd

if test -f "/usr/bin/python"; then
    echo "Not making python symlink"
else
    sudo ln -sf $(which python3) /usr/bin/python
fi
