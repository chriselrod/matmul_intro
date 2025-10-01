import MatMul;
import boost.ut;
import std;

using namespace boost::ut;

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

void initialize_matrix(std::vector<double> &matrix, long size,
                       std::mt19937 &gen) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (long i = 0; i < size; ++i) {
    matrix[i] = dist(gen);
  }
}

bool matrices_equal(const std::vector<double> &A, const std::vector<double> &B,
                    double tolerance = 1e-12) {
  if (A.size() != B.size())
    return false;

  for (std::size_t i = 0; i < A.size(); ++i) {
    if (std::abs(A[i] - B[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

int main() {
  std::mt19937 gen(42); // Fixed seed for reproducibility

  "48x48x48 matrix multiplication"_test = [] {
    std::mt19937 gen(42);
    constexpr long M = 72, N = 72, K = 72;

    std::vector<double> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    initialize_matrix(A, M * K, gen);
    initialize_matrix(B, K * N, gen);

    naive_matmul(C_ref.data(), A.data(), B.data(), M, N, K);
    matmul(C_test.data(), A.data(), B.data(), M, N, K);

    expect(matrices_equal(C_test, C_ref)) << "48x48x48 matrices should match";
  };

  "96x96x96 matrix multiplication"_test = [] {
    std::mt19937 gen(123);
    constexpr long M = 144, N = 144, K = 144;

    std::vector<double> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    initialize_matrix(A, M * K, gen);
    initialize_matrix(B, K * N, gen);

    naive_matmul(C_ref.data(), A.data(), B.data(), M, N, K);
    matmul(C_test.data(), A.data(), B.data(), M, N, K);

    expect(matrices_equal(C_test, C_ref)) << "96x96x96 matrices should match";
  };

  "144x48x96 rectangular matrix multiplication"_test = [] {
    std::mt19937 gen(456);
    constexpr long M = 144, N =72, K = 144;

    std::vector<double> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    initialize_matrix(A, M * K, gen);
    initialize_matrix(B, K * N, gen);

    naive_matmul(C_ref.data(), A.data(), B.data(), M, N, K);
    matmul(C_test.data(), A.data(), B.data(), M, N, K);

    expect(matrices_equal(C_test, C_ref)) << "144x48x96 matrices should match";
  };

  "48x144x96 rectangular matrix multiplication"_test = [] {
    std::mt19937 gen(789);
    constexpr long M = 72, N = 144, K = 72;

    std::vector<double> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    initialize_matrix(A, M * K, gen);
    initialize_matrix(B, K * N, gen);

    naive_matmul(C_ref.data(), A.data(), B.data(), M, N, K);
    matmul(C_test.data(), A.data(), B.data(), M, N, K);

    expect(matrices_equal(C_test, C_ref)) << "48x144x96 matrices should match";
  };

  return 0;
}
