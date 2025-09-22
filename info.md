cd /Users/sachinsheoran/Documents/masters_curriculum/project/duckdb

# Create optimized build directory
mkdir build-fast
cd build-fast

# Configure for fast builds
cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_FLAGS="-O1 -g1" \
      -DBUILD_UNITTESTS=OFF \
      ..

# Build
ninja -j$(sysctl -n hw.ncpu)


# Use the below command to typecast a pointer to its type in the debugger
`-exec p *(duckdb::ClientContext*)0x0000616000001598`

<!-- # Run these commands before starting any build, otherwise you might get a build error.
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export MACOS_SDK_PATH=$(xcrun --show-sdk-path)
export CPATH="${MACOS_SDK_PATH}/usr/include:$CPATH" -->

# Update commands
# Set Homebrew LLVM as primary toolchain
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++

# Use Homebrew LLVM's C++ standard library
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/llvm/include/c++/v1"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib -Wl,-rpath,/opt/homebrew/opt/llvm/lib"

# Force use of LLVM's libc++
export CXXFLAGS="-stdlib=libc++ -I/opt/homebrew/opt/llvm/include/c++/v1"

# Prioritize Homebrew LLVM in PATH
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

# Remove or modify the system SDK path to come AFTER LLVM headers
export MACOS_SDK_PATH=$(xcrun --show-sdk-path)
# Don't add system includes to CPATH - let compiler handle this
# export CPATH="${MACOS_SDK_PATH}/usr/include:$CPATH"  # Comment this out

export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/llvm/include/c++/v1"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib -Wl,-rpath,/opt/homebrew/opt/llvm/lib"
export CXXFLAGS="-stdlib=libc++ -I/opt/homebrew/opt/llvm/include/c++/v1"
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export MACOS_SDK_PATH=$(xcrun --show-sdk-path)

Actual file is present at : /Users/sachinsheoran/Documents/masters_curriculum/project/duckdb/src/include/lingodb/runtime/helpers.h
include_directories path : /Users/sachinsheoran/Documents/masters_curriculum/project/duckdb/src/include
import : lingodb/runtime/helpers.h


## Lingodb

<!-- SQL TO MLIR -->
`./build/lingodb-release/sql-to-mlir test.sql dbdir/`