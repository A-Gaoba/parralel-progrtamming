#include <iostream>
#include <cmath>
#include <cassert>
#include <omp.h>
#include <iomanip>
#include <ctime>

#define GREEN_TEXT "\033[32m"
#define CYAN_TEXT "\033[36m"
#define RESET_COLOR "\033[0m"
//is a structure that holds information about the calculation result, execution time, and the number of threads used
struct CalcInfo
{
  double result, time;
  int countThreads;
};

//calculate acceleration.
double calculateAcceleration(double t1, double tp)
{
  if (tp < 0.00000000001)
    return 0;
  return t1 / tp;
}

//calculate efficiency.
double calculateEfficiency(double sp, double p)
{
  return sp / p;
}

// performs a serial calculation for a given value of n. It calculates a result by summing up a series of mathematical operations
CalcInfo calculateResult(long long n)
{
  CalcInfo info;
  double result = 0;
  clock_t t1, t2;
  double time;

  t1 = clock();
  for (long long i = 1; i <= n; ++i)
  {
    result += pow(i, 1.0 / 3.0) / ((i + 1) * sqrt(i));
  }
  t2 = clock();

  time = (t2 - t1) / (double)CLOCKS_PER_SEC;

  info.result = result;
  info.time = time;
  info.countThreads = 1;

  return info;
}

// his function performs a parallel calculation using OpenMP. The number of threads used is specified by the countThreads parameter
CalcInfo calculateResultParallel(long long n, int countThreads = 8)
{
  CalcInfo info;
  double result = 0;
  clock_t t1, t2;
  double time;

  omp_set_num_threads(countThreads);
  int actualThreads;

#pragma omp parallel
  {
    actualThreads = omp_get_num_threads();
    t1 = clock();

#pragma omp for reduction(+ : result)
    for (long long i = 1; i <= n; ++i)
    {
      result += pow(i, 1.0 / 3.0) / ((i + 1) * sqrt(i));
    }

    t2 = clock();
  }

  time = (t2 - t1) / (double)CLOCKS_PER_SEC;

  info.result = result;
  info.time = time;
  info.countThreads = actualThreads;

  return info;
}

// This function extends the parallel calculation by allowing the specification of a scheduling strategy (static, dynamic, or guided) and a chunk size.
CalcInfo calculateResultParallelSchedule(long long n, std::string schedule, int chunk, int countThreads = 8)
{
  CalcInfo info;
  double result = 0;
  clock_t t1, t2;
  double time;

  omp_set_num_threads(countThreads);
  int actualThreads;

#pragma omp parallel
  {
    actualThreads = omp_get_num_threads();

    t1 = clock();

    if (schedule == "static")
    {
#pragma omp for schedule(static, chunk) reduction(+ : result)
      for (long long i = 1; i <= n; ++i)
      {
        result += pow(i, 1.0 / 3.0) / ((i + 1) * sqrt(i));
      }
    }
    else if (schedule == "dynamic")
    {
#pragma omp for schedule(dynamic, chunk) reduction(+ : result)
      for (long long i = 1; i <= n; ++i)
      {
        result += pow(i, 1.0 / 3.0) / ((i + 1) * sqrt(i));
      }
    }
    else if (schedule == "guided")
    {
#pragma omp for schedule(guided, chunk) reduction(+ : result)
      for (long long i = 1; i <= n; ++i)
      {
        result += pow(i, 1.0 / 3.0) / ((i + 1) * sqrt(i));
      }
    }

    t2 = clock();
  }

  time = (t2 - t1) / (double)CLOCKS_PER_SEC;

  info.result = result;
  info.time = time;
  info.countThreads = actualThreads;

  return info;
}

void printTableHeader(const std::string &title)
{
  std::cout << "\n"
            << title << "\n";
  std::cout << "-------------------------------------------------------------------------------------------------------\n";
  std::cout << "| Label          | Result          | Time            | Count Threads   | Acceleration    | Efficiency  |\n";
  std::cout << "-------------------------------------------------------------------------------------------------------\n";
}

void printTableRow(const std::string &label, double result, double time, int countThreads, double acceleration, double efficiency)
{
  std::cout << "| " << std::setw(15) << std::left << label
            << "| " << std::setw(15) << std::left << result
            << "| ";

  // Color the time value
  if (label == "Using parallel")
  {
    std::cout << GREEN_TEXT;
  }
  else
  {
    std::cout << CYAN_TEXT;
  }

  std::cout << std::setw(15) << std::left << time << RESET_COLOR
            << "| " << std::setw(15) << std::left << countThreads
            << "| " << std::setw(15) << std::left << acceleration
            << "| " << std::setw(15) << std::left << efficiency
            << "|\n";
  std::cout << "-------------------------------------------------------------------------------------------------------\n";
}


int main()
{
  // Task 3: Serial and parallel calculations are compared for increasing values of n.
  printTableHeader("Task 3");
  for (long long n = 100; n < 100000000; n *= 10)
  {
    CalcInfo info = calculateResult(n);
    CalcInfo info_parallel = calculateResultParallel(n);
    assert(std::abs(info.result - info_parallel.result) < 0.00001); // checks if the results of the serial and parallel computations are close enough. If not, it would trigger an error

    printTableRow("Not parallel", info.result, info.time, 1, 0, 0);
    printTableRow("Using parallel", info_parallel.result, info_parallel.time, info_parallel.countThreads,
                  calculateAcceleration(info.time, info_parallel.time),
                  calculateEfficiency(calculateAcceleration(info.time, info_parallel.time), info_parallel.countThreads));
  }

  // Task 4: Serial and parallel calculations are compared for increasing values of n and varying numbers of threads.
  printTableHeader("Task 4");
  for (long long threads = 5, n = 1000; n < 10000000000; n *= 10, threads += 1)
  {
    CalcInfo info = calculateResult(n);
    CalcInfo info_parallel = calculateResultParallel(n, threads);
    assert(std::abs(info.result - info_parallel.result) < 0.00001);

    printTableRow("Not parallel", info.result, info.time, 1, 0, 0);
    printTableRow("Using parallel", info_parallel.result, info_parallel.time, info_parallel.countThreads,
                  calculateAcceleration(info.time, info_parallel.time),
                  calculateEfficiency(calculateAcceleration(info.time, info_parallel.time), info_parallel.countThreads));
  }

  // Task 6: Serial and parallel calculations are compared for fixed values of n and different execution times {317600000, 620000000, 1350000000}.
  printTableHeader("Task 6");
  long long n_values[] = {317600000, 620000000, 1350000000};
  int times[] = {10, 20, 40};

  for (int i = 0; i < 3; ++i)
  {
    printTableHeader("Time = " + std::to_string(times[i]));

    std::cout << "Not parallel:\n";
    CalcInfo info = calculateResult(n_values[i]);
    printTableRow("", info.result, info.time, 1, 0, 0);

    std::cout << "Using parallel:\n";
    CalcInfo info_parallel = calculateResultParallel(n_values[i]);
    printTableRow("", info_parallel.result, info_parallel.time, info_parallel.countThreads,
                  calculateAcceleration(info.time, info_parallel.time),
                  calculateEfficiency(calculateAcceleration(info.time, info_parallel.time), info_parallel.countThreads));
  }

  // Task 8: Serial and parallel calculations are compared for fixed values of n using different scheduling strategies and chunk sizes.
  printTableHeader("Task 8");
  const int n_schedule_values = 3;
  const int chunk_values[] = {1, 100, 10000};
  std::string schedule_values[] = {"static", "dynamic", "guided"};

  for (int i = 0; i < n_schedule_values; ++i)
  {
    for (int j = 0; j < n_schedule_values; ++j)
    {
      std::string schedule = schedule_values[i];
      int chunk = chunk_values[j];

      std::cout << "\nSchedule(" << schedule << ", " << chunk << "):\n";
      CalcInfo info_schedule = calculateResultParallelSchedule(317600000, schedule, chunk);
      printTableRow("", info_schedule.result, info_schedule.time, info_schedule.countThreads,
                    calculateAcceleration(info_schedule.time, info_schedule.time),
                    calculateEfficiency(calculateAcceleration(info_schedule.time, info_schedule.time), info_schedule.countThreads));
    }
  }

  return 0;
}
