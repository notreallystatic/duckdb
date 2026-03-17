echo "Creating build directory..."
mkdir -p build-fast

rm -rf build-fast/*

echo "Running cmake..."
cmake -S . -B build-fast/ -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_FLAGS="-O1 -g1" \
      -DBUILD_UNITTESTS=OFF 
      # -DCMAKE_BUILD_TYPE=Debug \

echo "Building with ninja..."
ninja -C build-fast/ -j$(sysctl -n hw.ncpu)