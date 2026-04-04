# dynein-cell-model

### Local

NOTE: Need to figure out better compilation steps in the future

```bash
export CXXFLAGS="-O3 -march=native -ffast-math -DEIGEN_NO_DEBUG -DNDEBUG"
cmake \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DHDF5_ROOT=/opt/homebrew/opt/hdf5 \
    ..
cmake --build .
```

```bash
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOpenMP_libomp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
    -DOpenMP_CXX_FLAGS="-O0 -g -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    ..
cmake --build .
```

### PACE

```bash
cmake -DOpenCV_DIR=~/installs/opencv-4.x/build/ -DCMAKE_CXX_FLAGS="-O3 -march=native -DNDEBUG" ..
```
