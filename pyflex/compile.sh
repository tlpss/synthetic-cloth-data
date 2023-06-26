cd PyFlex/bindings
rm -rf build
mkdir build
cd build

cmake -DPYBIND11_PYTHON_VERSION=3.10 -DCMAKE_BUILD_TYPE=Release ..
make -j