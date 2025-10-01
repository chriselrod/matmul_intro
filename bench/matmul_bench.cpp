import std;
import MatMul;
import Nanobench;

void naive_matmul(double *__restrict__ C, double *__restrict__ A,
                  double *__restrict__ B, long M, long N, long K) {
  for (long m = 0; m < M; ++m) {
    for (long n = 0; n < N; ++n) {
      C[m * N + n] = 0.0;
    }
    for (long k = 0; k < K; ++k) {
      for (long n = 0; n < N; ++n) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

void initialize_matrix(std::vector<double> &matrix, std::mt19937 &gen) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto &x : matrix)
    x = dist(gen);
}

int main() {
  std::mt19937 gen(42); // Fixed seed for reproducibility

  // Small matrices for initial comparison
  {
    constexpr long M = 72, N = 72, K = 72;
    std::vector<double> A(M * K), B(K * N), C(M * N);

    initialize_matrix(A, gen);
    initialize_matrix(B, gen);

    Bench bench;
    bench.title("Matrix Multiplication 72x72x72")
        .unit("matrix")
        .warmup(10)
        .epochIterations(100);

    bench.minEpochIterations(1000).run("naive_matmul", [&] {
      naive_matmul(C.data(), A.data(), B.data(), M, N, K);
      doNotOptimizeAway(C);
    });

    bench.minEpochIterations(1000).run("optimized_matmul", [&] {
      matmul(C.data(), A.data(), B.data(), M, N, K);
      doNotOptimizeAway(C);
    });
  }

  // Medium matrices
  {
    constexpr long M = 216, N = 216, K = 216;
    std::vector<double> A(M * K), B(K * N), C(M * N);

    initialize_matrix(A, gen);
    initialize_matrix(B, gen);

    Bench bench;
    bench.title("Matrix Multiplication 216x216x216")
        .unit("matrix")
        .warmup(5)
        .epochIterations(50);

    bench.minEpochIterations(1000).run("naive_matmul", [&] {
      naive_matmul(C.data(), A.data(), B.data(), M, N, K);
      doNotOptimizeAway(C);
    });

    bench.minEpochIterations(1000).run("optimized_matmul", [&] {
      matmul(C.data(), A.data(), B.data(), M, N, K);
      doNotOptimizeAway(C);
    });
  }

  // Larger matrices
  {
    constexpr long M = 144, N = 144, K = 144;
    std::vector<double> A(M * K), B(K * N), C(M * N);

    initialize_matrix(A, gen);
    initialize_matrix(B, gen);

    Bench bench;
    bench.title("Matrix Multiplication 144x144x144")
        .unit("matrix")
        .warmup(3)
        .epochIterations(20);

    bench.minEpochIterations(1000).run("naive_matmul", [&] {
      naive_matmul(C.data(), A.data(), B.data(), M, N, K);
      doNotOptimizeAway(C);
    });

    bench.minEpochIterations(1000).run("optimized_matmul", [&] {
      matmul(C.data(), A.data(), B.data(), M, N, K);
      doNotOptimizeAway(C);
    });
  }

  // Rectangular matrices
  {
    constexpr long M = 144, N = 72, K = 216;
    std::vector<double> A(M * K), B(K * N), C(M * N);

    initialize_matrix(A, gen);
    initialize_matrix(B, gen);

    Bench bench;
    bench.title("Matrix Multiplication 144x72x216 (rectangular)")
        .unit("matrix")
        .warmup(5)
        .epochIterations(30);

    bench.minEpochIterations(1000).run("naive_matmul", [&] {
      naive_matmul(C.data(), A.data(), B.data(), M, N, K);
      doNotOptimizeAway(C);
    });

    bench.minEpochIterations(1000).run("optimized_matmul", [&] {
      matmul(C.data(), A.data(), B.data(), M, N, K);
      doNotOptimizeAway(C);
    });
  }

  return 0;
}
