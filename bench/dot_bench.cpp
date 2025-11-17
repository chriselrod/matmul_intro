import Nanobench;
import std;
import DotProduct;


auto initialize_vec(int vector_size, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  auto vec = std::vector<float>(vector_size);
  for (int i = 0; i < vector_size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

auto dot_product_clang_vectorized(const std::vector<float> &vec1, const std::vector<float> &vec2) {
  if (vec1.size() != vec2.size()) __builtin_trap();
  float sum = 0.0;
  for (int i = 0; i < vec1.size(); ++i) {
#pragma clang fp reassociate(on)
    sum += vec1[i] * vec2[i];
  }
  // for (auto x : vec){
  //   #pragma clang fp reassociate(on)
  //   sum += x;
  // }
  return sum;
}

int main() {
  std::mt19937 gen(42); // Fixed seed for reproducibility

  // Small vectors for initial comparison
  {
    constexpr int M = 512;
    std::vector<float> V1 = initialize_vec(M, gen);
    std::vector<float> V2 = initialize_vec(M, gen);

    Bench bench;
    bench.title("Summation M=512").unit("dots").warmup(1024).epochIterations(4096);

    auto my_sum = dot_product_vector(V1, V2);
    std::print("Diffs: Hand-optimized - Clang-Vectorized: {}\n",
               my_sum - dot_product_clang_vectorized(V1, V2));
    std::print("Diffs: Hand-optimized - std::reduce: {}\n",
               my_sum - std::inner_product(V1.begin(), V1.end(), V2.begin(), 0.0));

    bench.run("Clang-Vectorized", [&] {
      auto s = dot_product_clang_vectorized(V1, V2);
      doNotOptimizeAway(s);
    });
    bench.run("std::inner_product", [&] {
      auto s = std::inner_product(V1.begin(), V1.end(), V2.begin(), 0.0);
      doNotOptimizeAway(s);
    });
    bench.run("Hand-Optimized", [&] {
      auto s = dot_product_vector(V1, V2);
      doNotOptimizeAway(s);
    });
  }
  return 0;
}
