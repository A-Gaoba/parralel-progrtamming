// ParallelLab01.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <time.h>
#include <omp.h>
#include <math.h>

const int n = 2;

double func(int x)
{
    return pow(-1, x - 1) / (pow(x, 2) - x);
}

double time()
{
    clock_t t1, t2;
    double time;
    double sum = 0;
    t1 = clock();
    for (int i = n; i < 400000000; i++)
    {
        sum += func(i);
    }
    t2 = clock();

    return time = (t2 - t1) / (double)CLOCKS_PER_SEC;
    //return sum;
}

void get_treads()
{
    int numThreads2 = omp_get_num_threads();

    std::cout << "Threads count " << numThreads2 << "\n";
}

double time_parallel()
{
    clock_t t1, t2;
    double time;
    //double sum = 0;
    double pr = 0;

#pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            get_treads();
    }

    t1 = clock();

#pragma omp parallel for reduction (+ : pr)
    for (int i = n; i < 400000000; i++)
    {
        //get_treads();

        pr += func(i);
    }

    t2 = clock();
    
    //return pr;
    return time = (t2 - t1) / (double)CLOCKS_PER_SEC;
}

double time_parallel_schedule()
{
    clock_t t1, t2;
    double time;
    //double sum = 0;
    double pr = 0;

#pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            get_treads();
    }

    t1 = clock();

#pragma omp parallel for schedule (dynamic, 100)
    for (int i = n; i < 100000000; i++)
    {
        //get_treads();

        pr += func(i);
    }

    t2 = clock();

    //return pr;
    return time = (t2 - t1) / (double)CLOCKS_PER_SEC;
}

double time_parallelAuto()
{
    clock_t t1, t2;
    double time;
    //double sum = 0;
    double pr = 0;

#pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            get_treads();
    }

    t1 = clock();

    for (int i = n; i < 10000000; i++)
    {
        //get_treads();
        pr += func(i);
    }

    t2 = clock();

    //return pr;
    return time = (t2 - t1) / (double)CLOCKS_PER_SEC;
}

int main()
{

    int thread_count;
    std::cout << "Set threads count: ";
    std::cin >> thread_count;
    system("cls");

    std::cout << "Only 1 thread time " << time() << "\n";

    omp_set_num_threads(8);
    //get_treads();

    std::cout << "Only " << 8 << " thread time " << time_parallel() << "\n";

    omp_set_num_threads(1);

    std::cout << "Only auto" << " thread time " << time_parallel() << "\n";

    omp_set_num_threads(8);

    std::cout << "Only schedule thread " << 8 << ". thread time " << time_parallel_schedule() << "\n";
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
