
build/test/:
	CXXFLAGS="-stdlib=libc++ -march=native" CXX=clang++ cmake $(NINJAGEN) -S . -B build/test/ -DCMAKE_BUILD_TYPE=Debug

build/bench/:
	CXXFLAGS="-stdlib=libc++ -march=native" CXX=clang++ cmake $(NINJAGEN) -S . -B build/bench/ -DCMAKE_BUILD_TYPE=Release

test: build/test/
	cmake --build build/test/
	cmake --build build/test/ --target test

bench: build/bench/
	cmake --build build/bench/
	build/bench/matmul_bench


