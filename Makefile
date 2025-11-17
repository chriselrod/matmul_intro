
build/test/:
	CXXFLAGS="-stdlib=libc++ -march=native" CXX=clang++ cmake -G Ninja -S . -B build/test/ -DCMAKE_BUILD_TYPE=Debug

build/bench/:
	CXXFLAGS="-stdlib=libc++ -march=native" CXX=clang++ cmake -G Ninja -S . -B build/bench/ -DCMAKE_BUILD_TYPE=Release

test: build/test/
	cmake --build build/test/
	cmake --build build/test/ --target test

sum_bench: build/bench/
	cmake --build build/bench/
	build/bench/sum_bench

dot_product_bench: build/bench/
	cmake --build build/bench/
	build/bench/dot_product_bench

matmul_bench: build/bench/
	cmake --build build/bench/
	build/bench/matmul_bench

matrix_vector_bench: build/bench/
	cmake --build build/bench/
	build/bench/matrix_vector_bench

