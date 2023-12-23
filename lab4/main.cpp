#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <time.h>

#include "mpi.h"

#define FIRST_THREAD 0
#define num(row, col) ((col) + (row) * size_matrix)

struct CalcInfo
{
  double time;
  double *result;

  ~CalcInfo() { delete[] result; }
};

double getAccelerate(double t1, double tp)
{
  if (tp < 0.00000000001)
    return 0;
  return t1 / tp;
}

double getEffect(double sp, double p)
{
  return sp / p;
}

void printMatrix(int size, double *matrix)
{
  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      std::cout << std::setw(10) << std::setprecision(6) << std::fixed << matrix[i * size + j];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

CalcInfo multiMatrix(int size_matrix, double *matrix_a, double *matrix_b)
{
  clock_t t1, t2;
  double time;

  CalcInfo info;
  info.result = new double[size_matrix * size_matrix];

  t1 = clock();

  for (int i = 0; i < size_matrix; ++i)
  {
    for (int j = 0; j < size_matrix; ++j)
    {
      double sum = 0;
      for (int k = 0; k < size_matrix; ++k)
      {
        sum += matrix_a[num(i, k)] * matrix_b[num(k, j)];
      }
      info.result[num(i, j)] = sum;
    }
  }

  t2 = clock();
  time = (t2 - t1) / (double)CLOCKS_PER_SEC;

  info.time = time;

  return info;
}

double *sum_matrix(int size_matrix, double *matrix_a, double *matrix_b)
{
  double *result = new double[size_matrix * size_matrix];

  for (int i = 0; i < size_matrix; ++i)
  {
    for (int j = 0; j < size_matrix; ++j)
    {
      result[num(i, j)] = matrix_a[num(i, j)] + matrix_b[num(i, j)];
    }
  }

  return result;
}

double *multi_matrix(int size_matrix, double *matrix_a, double *matrix_b)
{
  double *result = new double[size_matrix * size_matrix];

  for (int i = 0; i < size_matrix; ++i)
  {
    for (int j = 0; j < size_matrix; ++j)
    {
      double sum = 0;
      for (int k = 0; k < size_matrix; ++k)
      {
        sum += matrix_a[num(i, k)] * matrix_b[num(k, j)];
      }
      result[num(i, j)] = sum;
    }
  }

  return result;
}

double *multiMpiMatrix(double *matrix_a, double *matrix_b, int size_matrix, int proc, int partition)
{
  double *result = new double[size_matrix * size_matrix];

  for (int i = proc * partition; i < proc * partition + partition; ++i)
  {
    if (i >= size_matrix)
    {
      break;
    }
    for (int j = 0; j < size_matrix; ++j)
    {
      double sum = 0;
      for (int k = 0; k < size_matrix; ++k)
      {
        sum += matrix_a[num(i, k)] * matrix_b[num(k, j)];
      }
      result[num(i, j)] = sum;
    }
  }

  return result;
}

double *sumMpiMatrix(double *matrix_a, double *matrix_b, int size_matrix, int proc, int partition)
{
  double *result = new double[size_matrix * size_matrix];

  for (int i = proc * partition; i < proc * partition + partition; ++i)
  {
    if (i >= size_matrix)
    {
      break;
    }
    for (int j = 0; j < size_matrix; ++j)
    {
      result[num(i, j)] = matrix_a[num(i, j)] + matrix_b[num(i, j)];
    }
  }

  return result;
}

int main(int argc, char **argv)
{
  double time_start, time_end;

  int l = 0;
  int size_matrix;
  int partition;

  int thread, thread_size;
  MPI_Status status;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &thread_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &thread);

  int processors[] = {1, 2, 4, 6, 8}; // Processors to test
  double durations[sizeof(processors) / sizeof(processors[0])];

  double *matrix_a;
  double *matrix_b;

  double *tmp_matrix_aa;
  double *par_matrix_aa;
  double *tmp_matrix_aab;
  double *par_matrix_aab;

  for (int p = 0; p < sizeof(processors) / sizeof(processors[0]); ++p)
  {
    int max_processors = processors[p];
    if (thread == FIRST_THREAD)
    {
      if (l == 0)
      {
        std::cout << "Please, enter L : ";
        std::cin >> l;
        std::cout << "\n";
      }

      size_matrix = 10 * l;
      partition = ceil((double)size_matrix / (thread_size - 1));

      matrix_a = new double[size_matrix * size_matrix];
      matrix_b = new double[size_matrix * size_matrix];

      for (int i = 0; i < size_matrix * size_matrix; ++i)
      {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(-0.5, 0.5);

        matrix_a[i] = dist(gen);
        matrix_b[i] = dist(gen);
      }

      time_start = MPI_Wtime();

      for (int to_thread = 1; to_thread < thread_size; ++to_thread)
      {
        MPI_Send(&size_matrix, 1, MPI_INT, to_thread, 0, MPI_COMM_WORLD);
        MPI_Send(matrix_a, (size_matrix * size_matrix), MPI_DOUBLE, to_thread, 1, MPI_COMM_WORLD);
        MPI_Send(matrix_b, (size_matrix * size_matrix), MPI_DOUBLE, to_thread, 2, MPI_COMM_WORLD);
      }

      tmp_matrix_aa = new double[size_matrix * size_matrix];
      par_matrix_aa = new double[size_matrix * size_matrix];

      for (int to_thread = 1; to_thread < thread_size; ++to_thread)
      {
        MPI_Recv(tmp_matrix_aa, (size_matrix * size_matrix), MPI_DOUBLE, to_thread, 3, MPI_COMM_WORLD, &status);
        for (int i = (to_thread - 1) * partition; i < (to_thread - 1) * partition + partition; ++i)
        {
          if (i >= size_matrix)
          {
            break;
          }
          for (int j = 0; j < size_matrix; ++j)
          {
            par_matrix_aa[num(i, j)] = tmp_matrix_aa[num(i, j)];
          }
        }
      }

      for (int to_thread = 1; to_thread < thread_size; ++to_thread)
      {
        MPI_Send(par_matrix_aa, (size_matrix * size_matrix), MPI_DOUBLE, to_thread, 4, MPI_COMM_WORLD);
      }

      tmp_matrix_aab = new double[size_matrix * size_matrix];
      par_matrix_aab = new double[size_matrix * size_matrix];

      for (int to_thread = 1; to_thread < thread_size; ++to_thread)
      {
        MPI_Recv(tmp_matrix_aab, (size_matrix * size_matrix), MPI_DOUBLE, to_thread, 5, MPI_COMM_WORLD, &status);
        for (int i = (to_thread - 1) * partition; i < (to_thread - 1) * partition + partition; ++i)
        {
          if (i >= size_matrix)
          {
            break;
          }
          for (int j = 0; j < size_matrix; ++j)
          {
            par_matrix_aab[num(i, j)] = tmp_matrix_aab[num(i, j)];
          }
        }
      }

      time_end = MPI_Wtime();
      // std::cout << "\n\nWhen number of processore is : " << processors[p] << std::endl;
      // std::cout << "\nMatrix A:\n\n";
      // printMatrix(size_matrix, matrix_a);
      // std::cout << "\nMatrix B:\n\n";
      // printMatrix(size_matrix, matrix_b);

      clock_t t1, t2;
      double time;
      t1 = clock();

      double *matrix_aa = multi_matrix(size_matrix, matrix_a, matrix_a);
      double *matrix_aab = sum_matrix(size_matrix, matrix_aa, matrix_b);

      t2 = clock();
      time = (t2 - t1) / (double)CLOCKS_PER_SEC;

      // std::cout << "\nMatrix A*A:\n\n";
      // printMatrix(size_matrix, matrix_aa);
      // std::cout << "\n";

      // std::cout << "\nMatrix A*A + B:\n\n";
      // printMatrix(size_matrix, matrix_aab);

      // std::cout << "\nParallel Matrix A*A:\n\n";
      // printMatrix(size_matrix, par_matrix_aa);

      // std::cout << "\nParallel Matrix A*A + B:\n\n";
      // printMatrix(size_matrix, par_matrix_aab);

      std::cout << "When number of processore is : " << processors[p] << "\n\n";
      std::cout << "Time Sequential: " << time << "\n";
      std::cout << "Time Parralel: " << (time_end - time_start) << "\n";
      double accelerate = getAccelerate(time, (time_end - time_start));
      std::cout << "Accelerate: " << accelerate << "\n";
      std::cout << "Effection: " << getEffect(accelerate, thread_size) << "\n";
      std::cout << "--------------------------------\n";

      delete[] matrix_a;
      delete[] matrix_b;
      delete[] matrix_aa;
      delete[] matrix_aab;
      delete[] tmp_matrix_aa;
      delete[] par_matrix_aa;
      delete[] tmp_matrix_aab;
      delete[] par_matrix_aab;
    }
    else
    {
      MPI_Recv(&size_matrix, 1, MPI_INT, FIRST_THREAD, 0, MPI_COMM_WORLD, &status);

      matrix_a = new double[size_matrix * size_matrix];
      matrix_b = new double[size_matrix * size_matrix];

      MPI_Recv(matrix_a, (size_matrix * size_matrix), MPI_DOUBLE, FIRST_THREAD, 1, MPI_COMM_WORLD, &status);
      MPI_Recv(matrix_b, (size_matrix * size_matrix), MPI_DOUBLE, FIRST_THREAD, 2, MPI_COMM_WORLD, &status);

      partition = ceil((double)size_matrix / (thread_size - 1));
      tmp_matrix_aa = multiMpiMatrix(matrix_a, matrix_a, size_matrix, thread - 1, partition);

      MPI_Send(tmp_matrix_aa, (size_matrix * size_matrix), MPI_DOUBLE, FIRST_THREAD, 3, MPI_COMM_WORLD);

      par_matrix_aa = new double[size_matrix * size_matrix];
      MPI_Recv(par_matrix_aa, (size_matrix * size_matrix), MPI_DOUBLE, FIRST_THREAD, 4, MPI_COMM_WORLD, &status);

      tmp_matrix_aab = sumMpiMatrix(par_matrix_aa, matrix_b, size_matrix, thread - 1, partition);

      MPI_Send(tmp_matrix_aab, (size_matrix * size_matrix), MPI_DOUBLE, FIRST_THREAD, 5, MPI_COMM_WORLD);

      delete[] matrix_a;
      delete[] matrix_b;
    }
  }

  MPI_Finalize();

  return 0;
}
