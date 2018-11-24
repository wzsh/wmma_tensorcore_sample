/////////////////////////////////////////
// Calcuating Matrix A*B+C (CPU Version)
// Created by Wang Zong-Sheng
// 2018/10/18

#include<iostream>
#include<ctime>

#define A_ROW 3
#define A_COL 2
#define B_ROW 2
#define B_COL 3
#define C_ROW 2
#define C_COL 2

using namespace std;

template <typename T>
void matrix_add(const T *A, const T *B, unsigned int row, unsigned int col, T *R) {
	for (unsigned int c = 0; c < col; c++) {
		for (unsigned int r = 0; r < row; r++) {
			unsigned int i = c*row + r;
			R[i] = A[i] + B[i];
		}
	}
}

template <typename T>
void matrix_mul(const T *A, unsigned int a_row, unsigned int a_col, const T *B, unsigned int b_row, unsigned int b_col, T *R) {
	memset(R, 0, a_col*b_row*sizeof(T));
	for (unsigned int c = 0; c < a_col; c++) {
		for (unsigned int r = 0; r < b_row; r++) {
			unsigned int index = c * b_row + r;
			for (unsigned int i = 0; i < a_row; i++) {
				R[index] += A[c*a_row + i] * B[i*b_row + r];
			}
		}
	}
}

template <typename T>
void print_matrix(T *M, unsigned int row, unsigned int col) {
	for (unsigned int c = 0; c < col; c++) {
		for (unsigned int r = 0; r < row; r++) {
			cout << M[c*row + r] << ", ";
		}
		cout << endl;
	}
}

int main(void) {
	//clock_t start_timer = clock();
	const int A[A_ROW*A_COL] = { 1, 0, -3,
						  -2, 4,  1};
	const int B[B_ROW*B_COL] = { 2, -1,
						   3,  0,
						  -5,  2};
	const int C[C_ROW*C_COL] = { 3, -1,
						  -2,  2};
	int AB[A_COL*B_ROW], R[C_ROW*C_COL];

	matrix_mul<int>(A, A_ROW, A_COL, B, B_ROW, B_COL, AB);
	matrix_add<int>(AB, C, C_ROW, C_COL, R);

	//clock_t stop_timer = clock();
	//double duration = double(stop_timer - start_timer);

	// for printing results
	cout << "A = " << endl;
	print_matrix<const int>(A, A_ROW, A_COL);

	cout << endl << "B = " << endl;
	print_matrix<const int>(B, B_ROW, B_COL);

	cout << endl << "C = " << endl;
	print_matrix<const int>(C, C_ROW, C_COL);

	cout << endl << "Result:" << endl;
	cout << "A x B = " << endl;
	print_matrix<int>(AB, B_ROW, A_COL);

	cout << endl << "A x B + C = " << endl;
	print_matrix<int>(R, C_ROW, C_COL);

	//cout << "elapsed time = " << duration << "s" <<  endl;
	return 0;
}