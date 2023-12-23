#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>

#include <omp.h>
#include <time.h>

// Represents a structure that holds information about the computation
struct CalcInfo
{
  double time;      // Records the time taken for a computation
  int countThreads; // Stores the count of threads used in parallel computations
};

// Calculates the acceleration factor given two time values (t1 and tp).
double getAccelerate(double t1, double tp)
{
  if (tp < 0.00000000001)
    return 0;
  return t1 / tp;
}

// Computes the effect or efficiency given speedup (sp) and processors/threads count (p).
double getEffect(double sp, double p)
{
  return sp / p;
}

// Calculates the theoretical peak performance given processors (p), operations per cycle (n), and clock speed (v).
double getRpeak(double p, double n, double v)
{
  return p * n * v;
}

// Calculates the real performance based on actual computations performed given operations (q), matrix size (n), and time taken (t).
double getRreal(double q, double n, double t)
{
  return q * (2 * pow(n, 3) - pow(n, 2)) / t;
}

// Fills a square matrix of given size with random values.
template <int size>
void fillingMatrix(float (&matrix)[size][size])
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-0.5, 0.5);

  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      matrix[i][j] = dist(gen);
    }
  }
}

//  Prints a square matrix of given size to the console.
template <int size>
void printMatrix(float (&matrix)[size][size])
{
  std::cout << matrix[0][0] << "  " << matrix[0][size - 1] << "\n"
            << matrix[size - 1][0] << "  " << matrix[size - 1][size - 1] << "\n\n";
}

// Performs matrix multiplication sequentially for a specified number of repetitions (q).
template <int size>
CalcInfo multiMatrix(float (&matrix_a)[size][size], float (&matrix_b)[size][size], float (&result)[size][size], int q)
{
  clock_t t1, t2;
  double time;

  t1 = clock();

  for (int repeat = 0; repeat < q; ++repeat)
  {
    for (int i = 0; i < size; ++i)
    {
      for (int j = 0; j < size; ++j)
      {
        result[i][j] = 0;
        for (int k = 0; k < size; ++k)
        {
          result[i][j] += matrix_a[i][k] * matrix_b[k][j];
        }
      }
    }
  }

  t2 = clock();
  time = (t2 - t1) / (double)CLOCKS_PER_SEC;

  CalcInfo info;
  info.time = time;
  info.countThreads = 1;

  return info;
}

// Performs matrix multiplication using OpenMP-based parallelization with different algorithms (number_alg) and specified thread counts (countThreads).
template <int size>
CalcInfo multiMatrixParallel(int number_alg, float (&matrix_a)[size][size], float (&matrix_b)[size][size], float (&result)[size][size], int q, int countThreads = 8)
{
  clock_t t1, t2;
  double time;

  t1 = clock();

  omp_set_num_threads(countThreads);

#pragma omp parallel
  countThreads = omp_get_num_threads();

  if (number_alg == 1)
  {
#pragma omp parallel for
    for (int repeat = 0; repeat < q; ++repeat)
    {
      for (int i = 0; i < size; ++i)
      {
        for (int j = 0; j < size; ++j)
        {
          result[i][j] = 0;
          for (int k = 0; k < size; ++k)
          {
            result[i][j] += matrix_a[i][k] * matrix_b[k][j];
          }
        }
      }
    }
  }
  else
  {
    if (number_alg == 2)
    {
      for (int repeat = 0; repeat < q; ++repeat)
      {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
          for (int j = 0; j < size; ++j)
          {
            result[i][j] = 0;
            for (int k = 0; k < size; ++k)
            {
              result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
          }
        }
      }
    }
    else
    {
      for (int repeat = 0; repeat < q; ++repeat)
      {
        for (int i = 0; i < size; ++i)
        {
#pragma omp parallel for
          for (int j = 0; j < size; ++j)
          {
            result[i][j] = 0;
            for (int k = 0; k < size; ++k)
            {
              result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
          }
        }
      }
    }
  }

  t2 = clock();
  time = (t2 - t1) / (double)CLOCKS_PER_SEC;

  CalcInfo info;
  info.time = time;
  info.countThreads = countThreads;

  return info;
}

// Similar to multiMatrix, but optimized for triangular matrices.
template <int size>
CalcInfo multiTriangleMatrix(float (&matrix_a)[size][size], float (&matrix_b)[size][size], float (&result)[size][size], int q)
{
  clock_t t1, t2;
  double time;

  t1 = clock();

  for (int repeat = 0; repeat < q; ++repeat)
  {
    for (int i = 0; i < size; ++i)
    {
      for (int j = i; j < size; ++j)
      {
        result[i][j] = 0;
        for (int k = 0; k < size; ++k)
        {
          result[i][j] += matrix_a[i][k] * matrix_b[k][j];
        }
      }
    }
  }

  t2 = clock();
  time = (t2 - t1) / (double)CLOCKS_PER_SEC;

  CalcInfo info;
  info.time = time;
  info.countThreads = 1;

  return info;
}

// Similar to multiMatrixParallel, but optimized for triangular matrices.
template <int size>
CalcInfo multiTriangleMatrixParallel(int number_alg, float (&matrix_a)[size][size], float (&matrix_b)[size][size], float (&result)[size][size], int q, int countThreads = 8)
{
  clock_t t1, t2;
  double time;

  t1 = clock();

  omp_set_num_threads(countThreads);

#pragma omp parallel
  countThreads = omp_get_num_threads();

  if (number_alg == 1)
  {
#pragma omp parallel for
    for (int repeat = 0; repeat < q; ++repeat)
    {
      for (int i = 0; i < size; ++i)
      {
        for (int j = i; j < size; ++j)
        {
          result[i][j] = 0;
          for (int k = 0; k < size; ++k)
          {
            result[i][j] += matrix_a[i][k] * matrix_b[k][j];
          }
        }
      }
    }
  }
  else
  {
    if (number_alg == 2)
    {
      for (int repeat = 0; repeat < q; ++repeat)
      {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
          for (int j = 0; j < size; ++j)
          {
            result[i][j] = 0;
            for (int k = 0; k < size; ++k)
            {
              result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
          }
        }
      }
    }
    else
    {
      for (int repeat = 0; repeat < q; ++repeat)
      {
        for (int i = 0; i < size; ++i)
        {
#pragma omp parallel for
          for (int j = 0; j < size; ++j)
          {
            result[i][j] = 0;
            for (int k = 0; k < size; ++k)
            {
              result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
          }
        }
      }
    }
  }

  t2 = clock();
  time = (t2 - t1) / (double)CLOCKS_PER_SEC;

  CalcInfo info;
  info.time = time;
  info.countThreads = countThreads;

  return info;
}

void displayInTable(double accelerate, double effection, double performanceReal, double performanceRealParallel, double u1, double up)
{
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << std::left << std::setw(25) << "| Parameter"
            << "| " << std::setw(15) << "Value"
            << " |" << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << std::setw(25) << "| Accelerate"
            << "| " << std::setw(15) << accelerate << " |" << std::endl;
  std::cout << std::setw(25) << "| Effection"
            << "| " << std::setw(15) << effection << " |" << std::endl;
  std::cout << std::setw(25) << "| Real Performance"
            << "| " << std::setw(15) << performanceReal << " |" << std::endl;
  std::cout << std::setw(25) << "| Real Perf. Parallel"
            << "| " << std::setw(15) << performanceRealParallel << " |" << std::endl;
  std::cout << std::setw(25) << "| U1"
            << "| " << std::setw(15) << u1 << " |" << std::endl;
  std::cout << std::setw(25) << "| UP"
            << "| " << std::setw(15) << up << " |" << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
}

void displayParallelInfo(double time, double timepar, int countThreads, int n, int q)
{
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << std::left << std::setw(25) << "| Parameter"
            << "| " << std::setw(15) << "Value"
            << " |" << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << std::setw(25) << "| Time"
            << "| " << std::setw(15) << time << " |" << std::endl;
  std::cout << std::setw(25) << "| Time Parallel"
            << "| " << std::setw(15) << timepar << " |" << std::endl;
  std::cout << std::setw(25) << "| Count threads"
            << "| " << std::setw(15) << countThreads << " |" << std::endl;
  std::cout << std::setw(25) << "| N"
            << "| " << std::setw(15) << n << " |" << std::endl;
  std::cout << std::setw(25) << "| Number of repetitions"
            << "| " << std::setw(15) << q << " |" << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
}

int main()
{
  while (true)
  {
    std::cout << "\n\n\tMENU:\n1->Task 1-3\n2->Task 4\n3->Task 5-8, 10\n4->Task 9\n5->Task 11\n6->Task 12\n0->Exit\n\n->";

    int menu = 0;
    std::cin >> menu;

    if (menu == 1)
    {
      std::cout << "\n\tTask 1\n\n";
      {
        // Initializes three square matrices matrix_a, matrix_b, and matrix_ab, each of size 2x2.
        const int n = 2;
        float matrix_a[n][n];
        float matrix_b[n][n];
        float matrix_ab[n][n];

        fillingMatrix(matrix_a);
        fillingMatrix(matrix_b);

        std::cout << "\nMatrix A:\n\n";
        printMatrix(matrix_a);
        std::cout << "\nMatrix B:\n\n";
        printMatrix(matrix_b);

        int q = 1000;

        std::cout << "\nMatrix A*B:\n\n";
        CalcInfo info = multiMatrix(matrix_a, matrix_b, matrix_ab, q);
        printMatrix(matrix_ab);
        // std::cout << "Time:\t " << info.time << "\tN = " << n << "\tNumber of repetitions:\t" << q << "\n";

        std::cout << "\n\tTask 2\n\n";
        std::cout << "\nMatrix A*B Parallel (Algoritm 1):\n\n";
        CalcInfo info_parall_1 = multiMatrixParallel(1, matrix_a, matrix_b, matrix_ab, q);
        // printMatrix(matrix_ab);
        displayParallelInfo(info.time, info_parall_1.time, info_parall_1.countThreads, n, q);

        std::cout << "\nMatrix A*B Parallel (Algoritm 2):\n\n";
        CalcInfo info_parall_2 = multiMatrixParallel(2, matrix_a, matrix_b, matrix_ab, q);
        // printMatrix(matrix_ab);
        displayParallelInfo(info.time, info_parall_2.time, info_parall_2.countThreads, n, q);

        std::cout << "\nMatrix A*B Parallel (Algoritm 3):\n\n";
        CalcInfo info_parall_3 = multiMatrixParallel(3, matrix_a, matrix_b, matrix_ab, q);
        // printMatrix(matrix_ab);
        displayParallelInfo(info.time, info_parall_3.time, info_parall_3.countThreads, n, q);
      }
    }

    if (menu == 2)
    {
      std::cout << "\n\tTask 4\n\n";
      {
        const int n = 100;
        float matrix_a[n][n];
        float matrix_b[n][n];
        float matrix_ab[n][n];

        fillingMatrix(matrix_a);
        fillingMatrix(matrix_b);

        std::cout << "\nMatrix A:\n\n";
        printMatrix(matrix_a);
        std::cout << "\nMatrix B:\n\n";
        printMatrix(matrix_b);

        int q = 3000;

        std::cout << "\nMatrix A*B:\n\n";
        CalcInfo info = multiMatrix(matrix_a, matrix_b, matrix_ab, q);
        printMatrix(matrix_ab);
        // std::cout << "Time:\t " << info.time << "\tN = " << n << "\tNumber of repetitions:\t" << q << "\n";
        std::cout << "-------------------------------------+\n";
        std::cout << "| Item              |    Value       |\n";
        std::cout << "-------------------------------------+\n";
        std::cout << "| Time:             |    " << std::setw(12) << std::left << info.time << "|\n";
        std::cout << "| N                 |    " << std::setw(12) << std::left << n << "|\n";
        std::cout << "| Repetitions:      |    " << std::setw(12) << std::left << q << "|\n";
        std::cout << "-------------------------------------+\n";

        std::cout << "\nMatrix A*B Parallel (Algoritm 1):\n\n";
        CalcInfo info_parall_1 = multiMatrixParallel(1, matrix_a, matrix_b, matrix_ab, q);
        printMatrix(matrix_ab);
        displayParallelInfo(info.time, info_parall_1.time, info_parall_1.countThreads, n, q);

        std::cout << "\nMatrix A*B Parallel (Algoritm 2):\n\n";
        CalcInfo info_parall_2 = multiMatrixParallel(2, matrix_a, matrix_b, matrix_ab, q);
        printMatrix(matrix_ab);
        displayParallelInfo(info.time, info_parall_2.time, info_parall_2.countThreads, n, q);

        std::cout << "\nMatrix A*B Parallel (Algoritm 3):\n\n";
        CalcInfo info_parall_3 = multiMatrixParallel(3, matrix_a, matrix_b, matrix_ab, q);
        printMatrix(matrix_ab);
        displayParallelInfo(info.time, info_parall_3.time, info_parall_3.countThreads, n, q);
      }
    }

    if (menu == 3)
    {
      std::cout << "\n\tTask 5-8, 10\n\n";
      {
        const int n = 5;
        float matrix_a[n][n];
        float matrix_b[n][n];
        float matrix_ab[n][n];

        fillingMatrix(matrix_a);
        fillingMatrix(matrix_b);

        std::cout << "\nMatrix A:\n\n";
        printMatrix(matrix_a);
        std::cout << "\nMatrix B:\n\n";
        printMatrix(matrix_b);

        int q = 12000000;

        std::cout << "\nMatrix A*B:\n\n";
        CalcInfo info = multiMatrix(matrix_a, matrix_b, matrix_ab, q);
        printMatrix(matrix_ab);

        std::cout << "\nMatrix A*B Parallel (Algoritm 1):\n\n";
        CalcInfo info_parall = multiMatrixParallel(1, matrix_a, matrix_b, matrix_ab, q);
        printMatrix(matrix_ab);
        std::cout << "\nTask 6\n";
        displayParallelInfo(info.time, info_parall.time, info_parall.countThreads, n, q);

        double accelerate = getAccelerate(info.time, info_parall.time);
        double effection = getEffect(accelerate, info_parall.countThreads);
        double performanceReal = getRreal(q, n, info.time);
        double performanceRealParallel = getRreal(q, n, info_parall.time);
        double u1 = performanceReal / getRpeak(4, 8, 1.6);
        double up = performanceRealParallel / getRpeak(4, 8, 1.6);
        std::cout << "\nTask 7 and 10\n";
        displayInTable(accelerate, effection, performanceReal, performanceRealParallel, u1, up);
      }
    }

    if (menu == 4)
    {
      std::cout << "\n______________ Task 9 _____________________\n";
      std::cout << "+------------------------------------------+\n";
      double performancePeak1 = getRpeak(1, 8, 1.6);
      double performancePeakMulti = getRpeak(4, 8, 1.6);
      std::cout << "| Perfomance 1 Core:     | " << std::setw(15) << performancePeak1 << " |\n";
      std::cout << "+------------------------+-----------------+\n";
      std::cout << "| Perfomance Multi Core: | " << std::setw(15) << performancePeakMulti << " |\n";
      std::cout << "+------------------------------------------+\n";
    }

    if (menu == 5)
    {
      std::cout << "\n\tTask 11\n\n";
      {
        const int n = 100;
        float matrix_a[n][n];
        float matrix_b[n][n];
        float matrix_ab[n][n];

        fillingMatrix(matrix_a);
        fillingMatrix(matrix_b);

        std::cout << "\nMatrix A:\n\n";
        printMatrix(matrix_a);
        std::cout << "\nMatrix B:\n\n";
        printMatrix(matrix_b);

        int q = 1450;

        for (int p = 1; p < 7; ++p)
        {
          std::cout << "\nMatrix A*B:\n\n";
          CalcInfo info = multiMatrix(matrix_a, matrix_b, matrix_ab, q);
          printMatrix(matrix_ab);

          std::cout << "\nMatrix A*B Parallel (Algoritm 1):\n\n";
          CalcInfo info_parall = multiMatrixParallel(1, matrix_a, matrix_b, matrix_ab, q, p);
          // printMatrix(matrix_ab);

          std::cout << "When the number of threads equal : " << p << std::endl;
          displayParallelInfo(info.time, info_parall.time, info_parall.countThreads, n, q);
          double accelerate = getAccelerate(info.time, info_parall.time);
          double effection = getEffect(accelerate, info_parall.countThreads);
          std::cout << std::setw(25) << "| Accelerate"
                    << "| " << std::setw(15) << accelerate << " |" << std::endl;
          std::cout << std::setw(25) << "| Effection"
                    << "| " << std::setw(15) << effection << " |" << std::endl;
          std::cout << "+------------------------------------------+" << std::endl;
        }
      }
    }

    if (menu == 6)
    {

      std::cout << "\n\tTask 12 Triangle Matrix\n\n";
      {
        const int n = 100;
        float matrix_a[n][n];
        float matrix_b[n][n];
        float matrix_ab[n][n];

        fillingMatrix(matrix_a);
        fillingMatrix(matrix_b);

        std::cout << "\nMatrix A:\n\n";
        printMatrix(matrix_a);
        std::cout << "\nMatrix B:\n\n";
        printMatrix(matrix_b);

        int q = 1450;

        for (int p = 1; p < 7; ++p)
        {
          std::cout << "\nMatrix A*B:\n\n";
          CalcInfo info = multiTriangleMatrix(matrix_a, matrix_b, matrix_ab, q);
          printMatrix(matrix_ab);

          std::cout << "\nMatrix A*B Parallel (Algoritm 1):\n\n";
          CalcInfo info_parall = multiTriangleMatrixParallel(1, matrix_a, matrix_b, matrix_ab, q, p);
          printMatrix(matrix_ab);
          std::cout << "When the number of threads equal : " << p << std::endl;
          displayParallelInfo(info.time, info_parall.time, info_parall.countThreads, n, q);
          double accelerate = getAccelerate(info.time, info_parall.time);
          double effection = getEffect(accelerate, info_parall.countThreads);
          std::cout << std::setw(25) << "| Accelerate"
                    << "| " << std::setw(15) << accelerate << " |" << std::endl;
          std::cout << std::setw(25) << "| Effection"
                    << "| " << std::setw(15) << effection << " |" << std::endl;
          std::cout << "+------------------------------------------+" << std::endl;
        }
      }
    }

    if (menu == 0)
    {
      break;
    }
  }

  return 0;
}
