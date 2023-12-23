#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <iomanip>
#include "mpi.h"

struct CalcInfo
{
  double result, time;
  int countThreads;
};

struct CalculationData
{
  int N;
  int numProcessors;
  double executionTime;
  double acceleration;
  double efficiency;
};

// Calculates the sum of values in a vector.
double sumVector(const std::vector<double> &values)
{
  double sum = 0;
  for (const auto &val : values)
  {
    sum += val;
  }
  return sum;
}

// Calculates a part of the overall sum based on a range for a given processor.
double calculateSumPart(int threadSize, int procRank, int n)
{
  int partition = (n + threadSize - 1) / threadSize;
  double sum = 0;

  int start = procRank * partition + 1;
  int end = std::min(n, start + partition);

  for (int i = start; i <= end; ++i)
  {
    sum += pow(i, 1.0 / 3.0) / ((i + 1) * sqrt(i));
  }

  return sum;
}

// Sequentially calculates the sum of a series for benchmarking purposes.
CalcInfo calculateSequentialSum(int n)
{
  CalcInfo info;
  clock_t startTime, endTime;

  startTime = clock();
  double sum = 0;
  for (int i = 1; i <= n; ++i)
  {
    sum += pow(i, 1.0 / 3.0) / ((i + 1) * sqrt(i));
  }
  endTime = clock();

  info.result = sum;
  info.time = (endTime - startTime) / static_cast<double>(CLOCKS_PER_SEC);

  return info;
}

double calculateAcceleration(double t1, double tp)
{
  if (tp < 0.00000000001)
    return 0;
  return t1 / tp;
}

double calculateEfficiency(double speedup, double p)
{
  return speedup / p;
}

int main(int argc, char **argv)
{
  int n;
  int threadRank, threadSize;
  MPI_Status status;

  // Initializes MPI and gets the rank and size of the processes
  MPI_Init(&argc, &argv);

  // Gets the total number of processes (threadSize) and the rank of the current process (threadRank).
  MPI_Comm_size(MPI_COMM_WORLD, &threadSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &threadRank);

  if (threadRank == 0)
  {
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::setw(10) << "N" << std::setw(15) << "Processors" << std::setw(15) << "Time (sec)" << std::setw(15) << "Acceleration" << std::setw(15) << "Efficiency" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int numProcessors : {1, 2, 4, 6, 8})
    {
      int N1 = 100000;
      int N2 = 200000;

      n = N1;
      double timeStart = MPI_Wtime();
      while (true)
      {
        // Broadcast the value of 'n' to all processes
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate local sum for each process
        double localSum = calculateSumPart(threadSize, threadRank, n);
        std::vector<double> partialSums(threadSize);
        MPI_Gather(&localSum, 1, MPI_DOUBLE, partialSums.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (threadRank == 0)
        {
          double timeEnd = MPI_Wtime();
          double totalTime = timeEnd - timeStart;
          CalcInfo sequentialInfo = calculateSequentialSum(n);
          double accelerate = calculateAcceleration(sequentialInfo.time, totalTime);
          double efficiency = calculateEfficiency(accelerate, numProcessors);

          if (totalTime >= 15.0)
          {
            std::cout << std::setw(10) << n << std::setw(15) << numProcessors << std::setw(15) << totalTime << std::setw(15) << accelerate << std::setw(15) << efficiency << std::endl;
            break;
          }
          else
          {
            N1 *= 2;
          }
        }
      }

      n = N2;
      timeStart = MPI_Wtime();
      while (true)
      {
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        double localSum = calculateSumPart(threadSize, threadRank, n);
        std::vector<double> partialSums(threadSize);
        MPI_Gather(&localSum, 1, MPI_DOUBLE, partialSums.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (threadRank == 0)
        {
          double timeEnd = MPI_Wtime();
          double totalTime = timeEnd - timeStart;
          CalcInfo sequentialInfo = calculateSequentialSum(n);
          double accelerate = calculateAcceleration(sequentialInfo.time, totalTime);
          double efficiency = calculateEfficiency(accelerate, numProcessors);

          if (numProcessors == 8 && totalTime >= 10.0 && totalTime <= 20.0) // Adjusted condition for N2 with 8 processors
          {
            std::cout << std::setw(10) << n << std::setw(15) << numProcessors << std::setw(15) << totalTime << std::setw(15) << accelerate << std::setw(15) << efficiency << std::endl;
            std::cout << std::string(70, '-') << std::endl;

            break;
          }
          else if (totalTime >= 30.0 && totalTime <= 40.0 && numProcessors != 8)
          {
            std::cout << std::setw(10) << n << std::setw(15) << numProcessors << std::setw(15) << totalTime << std::setw(15) << accelerate << std::setw(15) << efficiency << std::endl;
            std::cout << std::string(70, '-') << std::endl;
            break;
          }
          else
          {
            N2 *= 2;
          }
        }
      }
    }
  }

  MPI_Finalize();

  return 0;
}
