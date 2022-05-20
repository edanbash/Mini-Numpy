#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows < 0 || cols < 0) {
        PyErr_SetString(PyExc_ValueError, "Rows or columns are nonpositive.");
        return -1;
    }

    matrix *new = (matrix *) malloc(sizeof(matrix));
    if (new == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
        return -1;
    }

    new->rows = rows;
    new->cols = cols;
    new->ref_cnt = (int *) malloc(sizeof(int));
    if (new->ref_cnt == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
        return -1;
    }
    *new->ref_cnt = 1;

    double *data_p = (double *) calloc(rows * cols, sizeof(double));
    if (data_p == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
        return -1;
    }

    new->data = (double **) malloc(sizeof(double *) * rows);
    if (new->data == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
        return -1;
    }

    for (int r = 0; r < rows; r++) {
        new->data[r] = data_p + r * cols;
    }
    
    if (rows == 1 || cols ==1) {
        new->is_1d = 1;
    } else {
        new->is_1d = 0;
    }
    new->parent = NULL; 
    *mat = new;
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    matrix *new = (matrix *) malloc(sizeof(matrix));
    if (new == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
        return -1;
    }

    new->rows = rows;
    new->cols = cols;
    new->ref_cnt = from->ref_cnt;
    *new->ref_cnt += 1;
    new->data = (double **) malloc(sizeof(double *) * rows);
    if (new->data == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
        return -1;
    }

    *new->data = from->data[row_offset];
    for (int r = row_offset; r < rows + row_offset; r++) {
        new->data[r - row_offset] = from->data[r] + col_offset;
    }
   
    if (rows == 1 || cols == 1) {
        new->is_1d = 1;
    } else {
        new->is_1d = 0;
    }

    new->parent = from;
    *mat = new;
    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    if (mat != NULL) {
        if (*mat->ref_cnt == 1) {
            free(mat->data[0]);
            free(mat->ref_cnt);
        } else {
            *mat->ref_cnt -= 1;
        }
        free(mat->data);
        free(mat);    
    }
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    __m256d a = _mm256_set1_pd(val);
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols/4 * 4; c+=4) {
            _mm256_storeu_pd(&mat->data[r][c], a);
        }

        for(int c = mat->cols/4 * 4; c < mat->cols; c++) {
            mat->data[r][c] = val;
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols || result->rows != mat1->rows || result->cols != mat1->cols) {
        PyErr_SetString(PyExc_ValueError, "Rows or cols do not match");
        return -1;
    }

    int total_size = mat1->rows * mat1->cols;
    double *mat1_data = mat1->data[0];
    double *mat2_data = mat2->data[0];
    double *result_data = result->data[0];
    
    #pragma omp parallel for
    for (int i = 0; i < total_size/16 * 16; i+=16) {
        _mm256_storeu_pd(result_data + i, 
            _mm256_add_pd(_mm256_loadu_pd(mat1_data + i), _mm256_loadu_pd(mat2_data + i)));

        _mm256_storeu_pd(result_data + i + 4, 
            _mm256_add_pd(_mm256_loadu_pd(mat1_data + i + 4), _mm256_loadu_pd(mat2_data + i + 4)));

        _mm256_storeu_pd(result_data + i + 8, 
            _mm256_add_pd(_mm256_loadu_pd(mat1_data + i + 8), _mm256_loadu_pd(mat2_data + i + 8)));
        
        _mm256_storeu_pd(result_data + i + 12, 
            _mm256_add_pd(_mm256_loadu_pd(mat1_data + i + 12), _mm256_loadu_pd(mat2_data + i + 12)));
    }

    for (int i = total_size/16 * 16; i < total_size; i++) {
        result_data[i] = mat1_data[i] + mat2_data[i];
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols || result->rows != mat1->rows || result->cols != mat1->cols) {
        PyErr_SetString(PyExc_ValueError, "Rows or cols do not match");
        return -1;
    }

    int total_size = mat1->rows * mat1->cols;
    double *mat1_data = mat1->data[0];
    double *mat2_data = mat2->data[0];
    double *result_data = result->data[0];
    
    #pragma omp parallel for
    for (int i = 0; i < total_size/16 * 16; i+=16) {
        _mm256_storeu_pd(result_data + i, 
            _mm256_sub_pd(_mm256_loadu_pd(mat1_data + i), _mm256_loadu_pd(mat2_data + i)));

        _mm256_storeu_pd(result_data + i + 4, 
            _mm256_sub_pd(_mm256_loadu_pd(mat1_data + i + 4), _mm256_loadu_pd(mat2_data + i + 4)));

        _mm256_storeu_pd(result_data + i + 8, 
            _mm256_sub_pd(_mm256_loadu_pd(mat1_data + i + 8), _mm256_loadu_pd(mat2_data + i + 8)));
        
        _mm256_storeu_pd(result_data + i + 12, 
            _mm256_sub_pd(_mm256_loadu_pd(mat1_data + i + 12), _mm256_loadu_pd(mat2_data + i + 12)));
    }

    for (int i = total_size/16 * 16; i < total_size; i++) {
        result_data[i] = mat1_data[i] - mat2_data[i];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->data == NULL || mat2->data == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Matrix is null");
        return -1;
    }
    if (mat1->cols != mat2->rows || result->rows != mat1->rows || result->cols != mat2->cols) {
        PyErr_SetString(PyExc_ValueError, "Rows or cols is out of bounds");
        return -1;
    }

    double *mat1_data = mat1->data[0];
    double *mat2_data = mat2->data[0];
    double *result_data = result->data[0];
    double *temp = malloc(sizeof(double) * mat2->rows * mat2->cols);

    #pragma omp parallel for
    for (int k = 0; k < mat2->rows; k++) {
        for (int j = 0; j < mat2->cols/16 *16; j+=16) {
            temp[j*mat2->rows + k] = mat2_data[k*mat2->cols + j];
            temp[(j + 1)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 1];
            temp[(j + 2)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 2];
            temp[(j + 3)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 3];
            temp[(j + 4)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 4];
            temp[(j + 5)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 5];
            temp[(j + 6)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 6];
            temp[(j + 7)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 7];
            temp[(j + 8)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 8];
            temp[(j + 9)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 9];
            temp[(j + 10)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 10];
            temp[(j + 11)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 11];
            temp[(j + 12)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 12];
            temp[(j + 13)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 13];
            temp[(j + 14)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 14];
            temp[(j + 15)*mat2->rows + k] = mat2_data[k*mat2->cols + j + 15];
        }
        for (int j = mat2->cols/16 *16; j < mat2->cols; j++) {
            temp[j*mat2->rows + k] = mat2_data[k*mat2->cols + j];
        }
    }

    int dim = mat1->rows * mat2->cols;
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        int r_row = i/mat2->cols;
        int r_col = i%mat2->cols;
        __m256d c = _mm256_set1_pd(0);
        double c_flat = 0;
        for (int k = 0; k < mat1->cols/16 * 16; k+=16) {
            c = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + r_row*mat1->cols + k), 
                _mm256_loadu_pd(temp + r_col*mat1->cols + k), c);
            c = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + r_row*mat1->cols + k + 4), 
                _mm256_loadu_pd(temp + r_col*mat1->cols + k + 4), c);
            c = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + r_row*mat1->cols + k + 8), 
                _mm256_loadu_pd(temp + r_col*mat1->cols + k + 8), c);
            c = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + r_row*mat1->cols + k + 12), 
                _mm256_loadu_pd(temp + r_col*mat1->cols + k + 12), c);
        }
        c_flat = c[0] + c[1] + c[2] + c[3];
        for (int k = mat1->cols/16 * 16; k < mat1->cols; k++) {
            c_flat += mat1_data[r_row*mat1->cols + k] * temp[r_col*mat1->cols + k];
        }
        result_data[i] = c_flat;
    }
    free(temp);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    if (mat->rows != mat->cols || pow < 0 || result->rows != mat->rows || result->cols != mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Matrix is not square or pow is negative");
        return -1;
    }
    
    //initialize result to identity
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            if (r == c) {
                result->data[r][c] = 1;
            } else {
                result->data[r][c] = 0;
            }
        }
    }

    if (pow == 0) return 0;
    else if (pow == 1) {
        double *mat_data = mat->data[0];
        double *result_data = result->data[0];

        for (int i = 0; i < mat->rows * mat->cols; i++) {
            result_data[i] = mat_data[i];
        }
        return 0;
    }

    matrix *copy1 = (matrix *) malloc(sizeof(matrix));
    int alloc_failed1 = allocate_matrix(&copy1, mat->rows, mat->cols);
    if (alloc_failed1) return -1;

    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            copy1->data[i][j] = mat->data[i][j];
        }
    }

    matrix *copy2 = (matrix *) malloc(sizeof(matrix));
    int alloc_failed2= allocate_matrix(&copy2, mat->rows, mat->cols);
    if (alloc_failed2) return -1;

    int pow_length = 0;
    int pow_temp = pow;
    while (pow_temp > 0) {
        pow_length++;
        pow_temp = pow_temp >> 1;
    }

    for (int i = pow_length - 1; i > 0; i--) {
        int mask = pow & (1 << (i - 1));
        mul_matrix(copy2, copy1, copy1);
        if (mask) {
            matrix *temp = copy1;
            copy1 = copy2;
            copy2 = temp;
            mul_matrix(copy2, copy1, mat);
        }
        matrix *temp = copy1;
        copy1 = copy2;
        copy2 = temp;
    }

    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result->data[i][j] = copy1->data[i][j];
        }
    }
    deallocate_matrix(copy1);
    deallocate_matrix(copy2);

    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    if (result->cols != mat->cols || result->rows != mat->rows || result->rows != mat->rows || result->cols != mat->cols) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error occured");
        return -1;
    }

    int total_size = mat->rows * mat->cols;
    double *mat_data = mat->data[0];
    double *result_data = result->data[0];
    
    #pragma omp parallel for
    for (int i = 0; i < total_size/16 * 16; i+=16) {
        _mm256_storeu_pd(result_data + i, 
            _mm256_mul_pd(_mm256_loadu_pd(mat_data + i), _mm256_set1_pd (-1)));

        _mm256_storeu_pd(result_data + i + 4, 
            _mm256_mul_pd(_mm256_loadu_pd(mat_data + i + 4), _mm256_set1_pd (-1)));

        _mm256_storeu_pd(result_data + i + 8, 
            _mm256_mul_pd(_mm256_loadu_pd(mat_data + i + 8), _mm256_set1_pd (-1)));
        
        _mm256_storeu_pd(result_data + i + 12, 
            _mm256_mul_pd(_mm256_loadu_pd(mat_data + i + 12), _mm256_set1_pd (-1)));
    }

    for (int i = total_size/16 * 16; i < total_size; i++) {
        result_data[i] = mat_data[i] * -1;
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    if (result->cols != mat->cols || result->rows != mat->rows || result->rows != mat->rows || result->cols != mat->cols) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error occured");
        return -1;
    }

    int total_size = mat->rows * mat->cols;
    double *mat_data = mat->data[0];
    double *result_data = result->data[0];
    
    #pragma omp parallel for
    for (int i = 0; i < total_size/16 * 16; i+=16) {
        __m256d mat_1 = _mm256_loadu_pd(mat_data + i);
        _mm256_storeu_pd(result_data + i, 
            _mm256_max_pd(mat_1, _mm256_mul_pd(mat_1, _mm256_set1_pd (-1))));
            
        __m256d mat_2 = _mm256_loadu_pd(mat_data + i + 4);
        _mm256_storeu_pd(result_data + i + 4, 
            _mm256_max_pd(mat_2, _mm256_mul_pd(mat_2, _mm256_set1_pd (-1))));

        __m256d mat_3 = _mm256_loadu_pd(mat_data + i + 8);
        _mm256_storeu_pd(result_data + i + 8, 
            _mm256_max_pd(mat_3, _mm256_mul_pd(mat_3, _mm256_set1_pd (-1))));
        
        __m256d mat_4 = _mm256_loadu_pd(mat_data + i + 12);
        _mm256_storeu_pd(result_data + i + 12, 
            _mm256_max_pd(mat_4, _mm256_mul_pd(mat_4, _mm256_set1_pd (-1))));
    }

    for (int i = total_size/16 * 16; i < total_size; i++) {
        if (mat_data[i] < 0) {
        result_data[i] = mat_data[i] * -1;
        } else {
            result_data[i] = mat_data[i];
        }
    }
    return 0;
}

