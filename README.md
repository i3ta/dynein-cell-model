# dynein-cell-model

### Local

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
