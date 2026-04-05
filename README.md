# dynein-cell-model

## Dependencies

### System Packages (Required)

Install these before building:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y libopencv-dev libhdf5-dev
```

**macOS (Homebrew):**
```bash
brew install opencv hdf5
```

### Finding HDF5 Paths (macOS)

If CMake can't find HDF5 automatically, you may need to specify the paths:

```bash
# Find HDF5 paths
brew --prefix hdf5
# Example output: /opt/homebrew/opt/hdf5

# Find the cmake config directory
ls /opt/homebrew/opt/hdf5/lib/cmake/
# Example output: hdf5-1.14.6
```

Then configure with:
```bash
mkdir build && cd build
cmake .. -DHDF5_DIR=/opt/homebrew/opt/hdf5/lib/cmake/hdf5-1.14.6
cmake --build . -j4
```

### Build

```bash
# Create build directory
mkdir build && cd build

# Configure (uses FetchContent for most dependencies)
cmake ..

# Build
cmake --build . -j4
```

### Optional: OpenMP (for parallelization)

OpenMP is auto-detected. On Linux with GCC, it will be enabled automatically.

**macOS:** OpenMP requires GCC (not Clang). To enable:
```bash
brew install gcc
export CC=/opt/homebrew/bin/gcc-15
export CXX=/opt/homebrew/bin/g++-15
cmake ..
```

**Linux:** Install GCC if not present:
```bash
sudo apt install build-essential
```

## Quick Start

```bash
# After building
./build/examples/run_model_metrics <config_file>
```

## Testing

```bash
# Run tests
cd build && ctest
```

## Project Structure

- `src/dynein_cell_model/` - Main cell model library
- `examples/` - Example executables
- `scripts/` - Python analysis scripts
- `cmake/` - CMake configuration files
