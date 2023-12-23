#include <iostream>
#include <mpi.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#define FIRST_THREAD 0


double getMaxVector(std::vector<double> values, int thread_size, int proc, int n)
{
    int partition = ceil((double)n / thread_size);
    double max_value = -1;
    
    for (int i = proc * partition; i < proc * partition + partition; ++i)
    {
        if (i >= n)
        {
            break;
        }
        if (values[i] > max_value)
        {
            max_value = values[i];
        }
    }

    return max_value;
}


double getAvgVector(std::vector<double> values, int thread_size, int proc, int n)
{
    int partition = ceil((double)n / thread_size);

    double sum = 0;
    for (int i = proc * partition; i < proc * partition + partition; ++i)
    {
        if (i >= n)
        {
            break;
        }
        sum += values[i];
    }

    return sum / values.size();
}


double sumVector(std::vector<double> value)
{
    double sum = 0;
    for (int i = 0; i < value.size(); ++i)
    {
        sum += value[i];
    }
    return sum;
}


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


int main(int argc, char** argv) 
{
    double time_start, time_end;
    double time_start_group1 = 100, time_end_group1;
    double time_start_group2 = 100, time_end_group2;

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    int vector_size = 0;

    std::vector<double> tmp_vector;

    std::vector<double> vector;
    if (rank == FIRST_THREAD)
    {
        std::cout << "Please, enter vector size: ";
        std::cin >> vector_size;

        vector.resize(vector_size);

        time_start = MPI_Wtime();

        for (int i = 0; i < vector_size; ++i)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(0, 500);

            vector[i] = dist(gen);
        }

        for (int to_thread = 1; to_thread < size; ++to_thread)
        {
            MPI_Send(vector.data(), vector.size(), MPI_DOUBLE, to_thread, 10, MPI_COMM_WORLD);
            MPI_Send(&vector_size, 1, MPI_INT, to_thread, 70, MPI_COMM_WORLD);

        }
    }
    else
    {
        MPI_Recv(&vector_size, 1, MPI_INT, FIRST_THREAD, 70, MPI_COMM_WORLD, &status);
        if (rank == 1)
        {
            vector.resize(vector_size);
            MPI_Recv(vector.data(), vector_size, MPI_DOUBLE, FIRST_THREAD, 10, MPI_COMM_WORLD, &status);
        }
    }

    int color = rank % 2;
    MPI_Comm group_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &group_comm);

    int group_rank, group_size;
    MPI_Comm_rank(group_comm, &group_rank);
    MPI_Comm_size(group_comm, &group_size);

    std::vector<double> local_vector(vector_size);

    if (group_rank == FIRST_THREAD)
    {
        if (color == 0) 
        {
            time_start_group1 = MPI_Wtime();
            MPI_Scatter(vector.data(), vector_size, MPI_DOUBLE, local_vector.data(), vector_size, MPI_DOUBLE, 0, group_comm);
        }
        else 
        {
            time_start_group2 = MPI_Wtime();
            MPI_Scatter(vector.data(), vector_size, MPI_DOUBLE, local_vector.data(), vector_size, MPI_DOUBLE, 0, group_comm);
        }
    }

    if (color == 0)
    {
        if (group_rank != FIRST_THREAD)
        {
            MPI_Recv(local_vector.data(), vector_size, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &status);
        }

        tmp_vector.push_back(getAvgVector(local_vector, group_size, group_rank, vector_size));
    }
    else
    {
        if (group_rank != FIRST_THREAD)
        {
            MPI_Recv(local_vector.data(), vector_size, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &status);
        }

        tmp_vector.push_back(getMaxVector(local_vector, group_size, group_rank, vector_size));
    }

    std::vector<double> group_result(group_size);

    if (color == 0)
    {
        MPI_Reduce(tmp_vector.data(), group_result.data(), group_size, MPI_DOUBLE, MPI_SUM, 0, group_comm);
    }
    else
    {
        MPI_Reduce(tmp_vector.data(), group_result.data(), group_size, MPI_DOUBLE, MPI_MAX, 0, group_comm);
    }

    if (group_rank == FIRST_THREAD)
    {
        if (color == 0) 
        {
            time_end_group1 = MPI_Wtime();
            std::cout << "\nAverage value in Group 1: " << group_result[0] << "\n" 
                << "Time group 1:\t " << (time_end_group1 - time_start_group1) << "\n";
        }
        else 
        {
            time_end_group2 = MPI_Wtime();
            std::cout << "\nMaximum value in Group 2: " << group_result[0] << "\n" 
                << "Time group 2:\t " << (time_end_group2 - time_start_group2) << "\n";
        }
    }

    if (rank == FIRST_THREAD)
    {
        time_end = MPI_Wtime();
        double time_par = (time_end - time_start);

        clock_t t1, t2;
        double time;
        t1 = clock();

        double avg_vec = std::accumulate(vector.begin(), vector.end(), 0.0) / vector.size();
        double max_vec = *std::max_element(vector.begin(), vector.end());

        t2 = clock();
        time = (t2 - t1) / (double)CLOCKS_PER_SEC;

        double accelerate = getAccelerate(time, time_par);

        std::cout << "\nTime all parallel:\t " << time_par << "\n"
            << "\nNot parallel Avg: " << avg_vec << "\n"
            << "Not parallel Max: " << max_vec << "\n"
            << "\nTime not parallel:\t " << time << "\n"
            << "\nAccelerate: " << accelerate << "\n"
            << "\nEffection: " << getEffect(accelerate, size) << "\n";
    }

    MPI_Finalize();

    return 0;
}
