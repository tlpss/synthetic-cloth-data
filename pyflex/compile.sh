cd PyFlex/bindings
rm -rf build
mkdir build
cd build
# Seuss
#if [[ $(hostname) = *"compute-0"* ]] || [[ $(hostname) = *"autobot-"* ]] || [[ $(hostname) = *"yertle"* ]]; then
#    export CUDA_BIN_PATH=/usr/local/cuda-9.1
#fi
cmake -DPYBIND11_PYTHON_VERSION=3.9 -DCMAKE_BUILD_TYPE=Release ..
# cmake -DPYBIND11_PYTHON_VERSION=3.6 -DCMAKE_BUILD_TYPE=Debug ..
make -j