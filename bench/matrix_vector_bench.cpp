import Nanobench;
import std;
import MatrixVector;

auto initialize_vec(int vector_size, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  auto vec = std::vector<float>(vector_size);
  for (int i = 0; i < vector_size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

void matrix_vector_clang(const std::vector<float> &mat,
                         const std::vector<float> &vec,
                         std::vector<float> &out) {
  std::ptrdiff_t rows = out.size(), cols = vec.size();
  if (mat.size() != rows * cols)
    __builtin_trap();
  for (std::ptrdiff_t row = 0; row < rows; ++row) {
    float acc = 0.0;
    for (std::ptrdiff_t col = 0; col < cols; ++col) {
#pragma clang fp reassociate(on)
      acc += mat[row * cols + col] * vec[col];
    }
    out[row] = acc;
  }
}

auto norm(const std::vector<float> &a, const std::vector<float> &b) -> float {
  std::ptrdiff_t len = a.size();
  if (b.size() != len)
    __builtin_trap();
  float diff = 0.0;
  for (std::ptrdiff_t i = 0; i < len; ++i) {
#pragma clang fp reassociate(on)
    float delta = a[i] - b[i];
    diff += delta * delta;
  }
  return std::sqrt(diff);
}

int main() {
  std::mt19937 gen(42); // Fixed seed for reproducibility

  // Small vectors for initial comparison
  {
    constexpr int M = 32;
    constexpr int N = 64;
    std::vector<float> mat = initialize_vec(M * N, gen);
    std::vector<float> vec = initialize_vec(N, gen);

    Bench bench;
    bench.title("Summation M=32,N=64")
        .unit("GEMV")
        .warmup(1024)
        .epochIterations(4096);

    std::vector<float> out(M);
    std::vector<float> out_ref(M);
    matrix_vector(mat, vec, out);
    matrix_vector_clang(mat, vec, out_ref);
    std::print("Norm: Hand-optimized - Clang-Vectorized: {}\n",
               norm(out, out_ref));

    bench.run("Clang-Vectorized", [&] {
      matrix_vector_clang(mat, vec, out);
      doNotOptimizeAway(out);
    });
    bench.run("Hand-Optimized", [&] {
      matrix_vector(mat, vec, out);
      doNotOptimizeAway(out);
    });
  }
  return 0;
}
