#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <iostream>
using namespace std;

#define BLOCK_SIZE 32

//функция ядра
__global__ void matrix_mult(const double* A, const double* B, double* C, int n)
{
	int ai = n * (blockDim.y * blockIdx.y + threadIdx.y);	// индекс начала строки матрицы A
	int bj = blockDim.x * blockIdx.x + threadIdx.x;			// индекс начала строки матрицы B
	double sum = 0;											// промежуточная переменная для вычиселний
	for (int k = 0; k < n; k++)
		sum += A[ai + k] * B[k * n + bj];					// вычисление произведения
	int index = n * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x; // индекс вычисляемого элемента матрицы C 
	C[index] = sum;											// заполнение массива результатми
}

// генерация матриц
double* generate_rand_matrix(int n, size_t size_matrix) {
	double* matrix = (double*)malloc(size_matrix);			// выделение памяти под массив
	for (int i = 0; i < n * n; i++) {
		matrix[i] = (double)rand() / (double)RAND_MAX;		// заполнение массива случайными числами
	}
	return matrix;											// возврат заполненной матрицы
}

// вывод матрицы в консоль (для проверки)
void print_matrix(double* matrix, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%4.1lf ", matrix[i * n + j]);
		}
		printf("\n");
	}
}

// функция для последовательного варианта умножения матриц
void matrix_mult_CPU(double* A, double* B, double* C, int n) {
	// реализация математического алгоритма умножения матриц
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}

// проверка результатов умножения
bool check_mult(double* C1, double* C2, int n) {
	double accuracy = 1.e-6;						//точность с которой будем производить проверку
	for (int i = 0; i < n * n; i++) {				// в цикле идем по всем ячейкам и 
		if (abs(C1[i] - C2[i]) >= accuracy)			// и проверяем если модуль разницы между значением полученным 
													// на ЦП и ГП больше либо равна нуля то тогда матрица посчитана неверно
			return false;
	}
	return true;									// иначе все пучком и матрица посчитана четко
}

int main(int argc, char* argv[])
{
	int N;										//размерность массива
	cout << "N=";
	
	while ( true)	// проверка корректного чтения размерности массива
	{
		cin >> N;
		if (N > 100 && N % 16 == 0)
		{
			break;
		}
		printf("You entered the dimension of the array incorrectly. Please repeat the input.\n");
	}

	// события начала и окончания времени
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	srand(time(NULL));
	size_t size_matrix = sizeof(double) * N * N;			// расширенная размерность массива на всякий случай

	// генерация массивов для работы на центральном процессоре
	double* A_CPU = generate_rand_matrix(N, size_matrix);
	double* B_CPU = generate_rand_matrix(N, size_matrix);
	double* C_CPU = (double*)malloc(size_matrix);			// матрица для копирования данных из памяти графического процессора
	double* C_seq_CPU = (double*)malloc(size_matrix);
	for (int i = 0; i < N * N; i++) {
		C_seq_CPU[i] = 0;
	}


	clock_t start_time = clock();                                                   // начало отсчета времени 
	matrix_mult_CPU(A_CPU, B_CPU, C_seq_CPU, N);									// расчет матричного произведения
	clock_t end_time = clock();                                                     // конечное время
	double search_time = (double)(end_time - start_time);                           // расчет затраченного времени
					
	
	printf("Time CPU: %f milliseconds\n", search_time);

	// выделение памяти на графическом процессоре
	double* A_GPU;
	cudaMalloc((void**)&A_GPU, size_matrix);
	double* B_GPU;
	cudaMalloc((void**)&B_GPU, size_matrix);
	double* C_GPU;
	cudaMalloc((void**)&C_GPU, size_matrix);

	// копирование данных в память графического процессора
	cudaMemcpy(A_GPU, A_CPU, size_matrix, cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, size_matrix, cudaMemcpyHostToDevice);

	// считаем число нитей и блоков для работы функции ядра
	dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = dim3(N / BLOCK_SIZE, N / BLOCK_SIZE);

	cudaEventRecord(start, 0);														// начало отсчета времени
	matrix_mult << <blocks, threads >> > (A_GPU, B_GPU, C_GPU, N);	                // работа функции ядра
	cudaEventRecord(stop, 0);														// окончание отсчета времени
	cudaEventSynchronize(stop);														// синхронизация

	// подсчет времени работы функции на графическом процессоре
	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, stop);
	printf("Time GPU: %f milliseconds\n", kernel_time);


	double S = search_time / kernel_time;
	printf("Acceleration: %f\n", S);

	// копирование результирующего массива из памяти графического процессора для последующей проверки
	cudaMemcpy(C_CPU, C_GPU, size_matrix, cudaMemcpyDeviceToHost);
	

	// проверка корректности вычисления
	if (check_mult(C_CPU, C_seq_CPU, N))
		printf("The multiplication results are correct.\n");
	else
		printf("Multiplication results are not correct.\n");
	//printf("Matrix:\n");
	//print_matrix(C_seq_CPU, N);
	// высвобождение памяти
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
	free(A_CPU);
	free(B_CPU);
	free(C_CPU);
	free(C_seq_CPU);

	return 0;
}