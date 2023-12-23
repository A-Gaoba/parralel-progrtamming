#include <iostream>
#include <iomanip>
#include <omp.h>

const int N = 100; // Matrix size
const int Q = 10;  // Number of repetitions for performance evaluation

// Function to initialize a matrix with random values in the range [-0.5, 0.5]
void initializeMatrix(double matrix[N][N])
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      matrix[i][j] = ((double)rand() / RAND_MAX) - 0.5;
    }
  }
}

// Function to multiply matrices (sequential version)
void multiplyMatricesSequential(double result[N][N], const double matrix1[N][N], const double matrix2[N][N])
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      result[i][j] = 0;
      for (int k = 0; k < N; ++k)
      {
        result[i][j] += matrix1[i][k] * matrix2[k][j];
      }
    }
  }
}

// Function to multiply matrices (parallel version using OpenMP)
void multiplyMatricesParallel(double result[N][N], const double matrix1[N][N], const double matrix2[N][N])
{
#pragma omp parallel for
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      result[i][j] = 0;
      for (int k = 0; k < N; ++k)
      {
        result[i][j] += matrix1[i][k] * matrix2[k][j];
      }
    }
  }
}

// Function to validate the results
bool validateResults(const double result1[N][N], const double result2[N][N])
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      if (std::abs(result1[i][j] - result2[i][j]) > 1e-6)
      {
        return false; // Results are not equal, validation failed
      }
    }
  }
  return true; // Results are equal, validation succeeded
}

int main()
{
  double matrix1[N][N], matrix2[N][N], resultSequential[N][N], resultParallel[N][N];

  // Step 1: Initialize matrices
  initializeMatrix(matrix1);
  initializeMatrix(matrix2);

  // Step 2: Perform sequential matrix multiplication and measure time
  double startTime = omp_get_wtime();
  multiplyMatricesSequential(resultSequential, matrix1, matrix2);
  double endTime = omp_get_wtime();
  std::cout << "Sequential Execution Time: " << endTime - startTime << " seconds\n";

  // Step 3: Perform parallel matrix multiplication and measure time
  startTime = omp_get_wtime();
  multiplyMatricesParallel(resultParallel, matrix1, matrix2);
  endTime = omp_get_wtime();
  std::cout << "Parallel Execution Time: " << endTime - startTime << " seconds\n";

  // Step 4: Validate the results
  if (validateResults(resultSequential, resultParallel))
  {
    std::cout << "Validation: Results are consistent.\n";
  }
  else
  {
    std::cout << "Validation: Results differ. Check the implementation.\n";
  }

  // Step 5: Evaluate performance for sequential program with different matrix sizes
  std::cout << "\nPerformance Evaluation for Sequential Program:\n";
  for (int size : {5, 10, 50, 100, 200, 500})
  {
    double avgTime = 0;
    for (int i = 0; i < Q; ++i)
    {
      initializeMatrix(matrix1);
      initializeMatrix(matrix2);
      startTime = omp_get_wtime();
      multiplyMatricesSequential(resultSequential, matrix1, matrix2);
      endTime = omp_get_wtime();
      avgTime += endTime - startTime;
    }
    avgTime /= Q;
    std::cout << "Matrix Size " << size << "x" << size << ": Average Execution Time = " << avgTime << " seconds\n";
  }

  // Step 6: Evaluate performance for parallel program with N=100 and Q repetitions
  std::cout << "\nPerformance Evaluation for Parallel Program (N=100):\n";
  for (int i = 0; i < Q; ++i)
  {
    initializeMatrix(matrix1);
    initializeMatrix(matrix2);
    startTime = omp_get_wtime();
    multiplyMatricesParallel(resultParallel, matrix1, matrix2);
    endTime = omp_get_wtime();
    std::cout << "Repetition " << i + 1 << ": Execution Time = " << endTime - startTime << " seconds\n";
  }

  // Step 7: Calculate floating-point operations and real efficiency for each N
  std::cout << "\nPerformance Analysis:\n";
  for (int size : {5, 10, 50, 100, 200, 500})
  {
    double avgTimeSequential = 0, avgTimeParallel = 0;
    for (int i = 0; i < Q; ++i)
    {
      initializeMatrix(matrix1);
      initializeMatrix(matrix2);

      // Sequential
      startTime = omp_get_wtime();
      multiplyMatricesSequential(resultSequential, matrix1, matrix2);
      endTime = omp_get_wtime();
      avgTimeSequential += endTime - startTime;

      // Parallel
      startTime = omp_get_wtime();
      multiplyMatricesParallel(resultParallel, matrix1, matrix2);
      endTime = omp_get_wtime();
      avgTimeParallel += endTime - startTime;
    }

    avgTimeSequential /= Q;
    avgTimeParallel /= Q;

    double flopCount = 2.0 * size * size * size; // Assuming N^3 operations in matrix multiplication

    double efficiency = flopCount / (avgTimeParallel * omp_get_max_threads()) / 1e9; // in GFLOP/s

    std::cout << "Matrix Size " << size << "x" << size << ":\n";
    std::cout << "  Sequential Time: " << avgTimeSequential << " seconds\n";
    std::cout << "  Parallel Time: " << avgTimeParallel << " seconds\n";
    std::cout << "  Floating-Point Operations: " << flopCount << " FLOP\n";
    std::cout << "  Real Efficiency: " << efficiency << " GFLOP/s\n";
    std::cout << "  Efficiency Ratio: " << efficiency / (2 * omp_get_max_threads()) << "\n\n";
  }

  return 0;
}
