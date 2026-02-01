# dynein-cell-model

### Local

```bash
cmake -B . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
cmake --build .
```

### PACE

```bash
cmake -DOpenCV_DIR=~/installs/opencv-4.x/build/ -DCMAKE_CXX_FLAGS="-O3 -march=native -DNDEBUG" ..
```
