import Nanobench;
import std;
import Sum;


auto initialize_vec(int vector_size, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  auto vec = std::vector<float>(vector_size);
  for (int i = 0; i < vector_size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

auto sum_clang_vectorized(const std::vector<float> &vec) {
  float sum = 0.0;
  for (int i = 0; i < vec.size(); ++i) {
#pragma clang fp reassociate(on)
    sum += vec[i];
  }
  // for (auto x : vec){
  //   #pragma clang fp reassociate(on)
  //   sum += x;
  // }
  return sum;
}

int main() {
  std::mt19937 gen(42); // Fixed seed for reproducibility

  // Small matrices for initial comparison
  {
    constexpr int M = 10000;
    std::vector<float> V = initialize_vec(M, gen);

    Bench bench;
    bench.title("Summation M=1000").unit("sum").warmup(10).epochIterations(100);

    auto my_sum = sum_vector(V);
    std::print("Diffs: Hand-optimized - Clang-Vectorized: {}\n",
               my_sum - sum_clang_vectorized(V));
    std::print("Diffs: Hand-optimized - std::reduce: {}\n",
               my_sum - std::reduce(V.begin(), V.end()));

    bench.run("Clang-Vectorized", [&] {
      auto s = sum_clang_vectorized(V);
      doNotOptimizeAway(s);
    });
    bench.run("std::reduce", [&] {
      auto s = std::reduce(V.begin(), V.end());
      doNotOptimizeAway(s);
    });
    bench.run("Hand-Optimized", [&] {
      auto s = sum_vector(V);
      doNotOptimizeAway(s);
    });
  }
  return 0;
}
