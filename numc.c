#include "numc.h"
#include <structmember.h>

PyTypeObject Matrix61cType;

/* Helper functions for initalization of matrices and vectors */

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols) {
  if (rows == 1 || cols == 1) {
    return PyTuple_Pack(1, PyLong_FromLong(rows * cols));
  } else {
    return PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
  }
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    rand_matrix(new_mat, seed, low, high);
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    else {
        fill_matrix(new_mat, val);
        ((Matrix61c *)self)->mat = new_mat;
        ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    }
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst) {
    if (rows * cols != PyList_Size(lst)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j, PyFloat_AsDouble(PyList_GetItem(lst, count)));
            count++;
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst) {
    int rows = PyList_Size(lst);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numc.Matrix with an empty list");
        return -1;
    }
    int cols;
    if (!PyList_Check(PyList_GetItem(lst, 0))) {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    } else {
        cols = PyList_Size(PyList_GetItem(lst, 0));
    }
    for (int i = 0; i < rows; i++) {
        if (!PyList_Check(PyList_GetItem(lst, i)) ||
                PyList_Size(PyList_GetItem(lst, i)) != cols) {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j,
                PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(lst, i), j)));
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrix61c_dealloc(Matrix61c *self) {
    deallocate_matrix(self->mat);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrix61c_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwds) {
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrix61c *self = (Matrix61c *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrix61c type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrix61c_init(PyObject *self, PyObject *args, PyObject *kwds) {
    /* Generate random matrices */
    if (kwds != NULL) {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (!PyBool_Check(rand)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (rand != Py_True) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0;
        double double_high = 1;
        unsigned int unsigned_seed = 0;

        if (low) {
            if (PyFloat_Check(low)) {
                double_low = PyFloat_AsDouble(low);
            } else if (PyLong_Check(low)) {
                double_low = PyLong_AsLong(low);
            }
        }

        if (high) {
            if (PyFloat_Check(high)) {
                double_high = PyFloat_AsDouble(high);
            } else if (PyLong_Check(high)) {
                double_high = PyLong_AsLong(high);
            }
        }

        if (double_low >= double_high) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        // Set seed if argument exists
        if (seed) {
            if (PyLong_Check(seed)) {
                unsigned_seed = PyLong_AsUnsignedLong(seed);
            }
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (PyArg_UnpackTuple(args, "args", 2, 2, &rows, &cols)) {
            if (rows && cols && PyLong_Check(rows) && PyLong_Check(cols)) {
                return init_rand(self, PyLong_AsLong(rows), PyLong_AsLong(cols), unsigned_seed, double_low,
                                 double_high);
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3)) {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3)
                || PyFloat_Check(arg3))) {
            if (PyLong_Check(arg3)) {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            } else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        } else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3)) {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        } else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        } else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrix61c_to_list(Matrix61c *self) {
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d) {  // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(get(self->mat, i, j)));
                count++;
            }
        }
    } else {  // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++) {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(get(self->mat, i, j)));
            }
        }
    }
    return py_lst;
}

PyObject *Matrix61c_class_to_list(Matrix61c *self, PyObject *args) {
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat)) {
        if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
            return NULL;
        }
        Matrix61c* mat61c = (Matrix61c*)mat;
        return Matrix61c_to_list(mat61c);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrix61c_class_methods[] = {
    {"to_list", (PyCFunction)Matrix61c_class_to_list, METH_VARARGS, "Returns a list representation of numc.Matrix"},
    {NULL, NULL, 0, NULL}
};

/*
 * Matrix61c string representation. For printing purposes.
 */
PyObject *Matrix61c_repr(PyObject *self) {
    PyObject *py_lst = Matrix61c_to_list((Matrix61c *)self);
    return PyObject_Repr(py_lst);
}

/* NUMBER METHODS */

/*
 * Add the second numc.Matrix (Matrix61c) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_add(Matrix61c* self, PyObject* args) {
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
        return NULL;
    }
    
    matrix* new_mat;
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols); 
    if(alloc_failed) return NULL;
    int add_failed = add_matrix(new_mat, self->mat, ((Matrix61c *)args)->mat);
    if(add_failed) return NULL;

    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->mat = new_mat;
    result->shape = get_shape(self->mat->rows, self->mat->cols);  
    return (PyObject *) result;
}

/*
 * Substract the second numc.Matrix (Matrix61c) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_sub(Matrix61c* self, PyObject* args) {
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
        return NULL;
    }
    
    matrix* new_mat;  
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if(alloc_failed) return NULL;
    int sub_failed = sub_matrix(new_mat, self->mat, ((Matrix61c *)args)->mat);
    if(sub_failed) return NULL;


    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);  
    result->mat = new_mat;
    result->shape = get_shape(self->mat->rows, self->mat->cols);  
    return (PyObject *) result;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrix61c_multiply(Matrix61c* self, PyObject *args) {
   if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
        return NULL;
    }
    
    matrix* new_mat;
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, ((Matrix61c *)args)->mat->cols);
    if(alloc_failed) return NULL;
    int mul_failed = mul_matrix(new_mat, self->mat, ((Matrix61c *)args)->mat);
    if(mul_failed) return NULL;
    
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);  
    result->mat = new_mat;
    result->shape = get_shape(self->mat->rows, ((Matrix61c *)args)->mat->cols);  
    return (PyObject *) result;
}

/*
 * Negates the given numc.Matrix.
 */
PyObject *Matrix61c_neg(Matrix61c* self) {
    matrix* new_mat;
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if(alloc_failed) return NULL;
    int neg_failed = neg_matrix(new_mat, self->mat);
    if(neg_failed) return NULL;

    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);  
    result->mat = new_mat;
    result->shape = get_shape(self->mat->rows, self->mat->cols);  
    return (PyObject *) result;
}

/*
 * Take the element-wise absolute value of this numc.Matrix.
 */
PyObject *Matrix61c_abs(Matrix61c *self) {    
    matrix* new_mat;
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if(alloc_failed) return NULL;
    int abs_failed = abs_matrix(new_mat, self->mat);
    if(abs_failed) return NULL;
    
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);  
    result->mat = new_mat;
    result->shape = get_shape(self->mat->rows, self->mat->cols);  
    return (PyObject *) result;
}

/*
 * Raise numc.Matrix (Matrix61c) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrix61c_pow(Matrix61c *self, PyObject *pow, PyObject *optional) {
    if (!PyLong_Check(pow)) {
        PyErr_SetString(PyExc_TypeError, "Power is not an int");
        return NULL;
    }

    matrix* new_mat;
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if(alloc_failed) return NULL;
    int pow_failed = pow_matrix(new_mat, self->mat, PyFloat_AsDouble(pow));
    if(pow_failed) return NULL;

    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);  
    result->mat = new_mat;
    result->shape = get_shape(self->mat->rows, self->mat->cols);  
    return (PyObject *) result;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrix61c_as_number = {
    .nb_add = (binaryfunc) Matrix61c_add,
    .nb_subtract = (binaryfunc) Matrix61c_sub,
    .nb_multiply = (binaryfunc) Matrix61c_multiply,
    .nb_absolute = (unaryfunc) Matrix61c_abs,
    .nb_power = (ternaryfunc) Matrix61c_pow,
    .nb_negative = (unaryfunc) Matrix61c_neg
};


/* INSTANCE METHODS */

/*
 * Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrix61c_set_value(Matrix61c *self, PyObject* args) {
    if(!PyTuple_Check(args)) {
        PyErr_SetString(PyExc_TypeError, "Args is not of right type or size.");
        return NULL;
    }

    PyObject *row = NULL;
    PyObject *col = NULL;
    PyObject *val = NULL;
    PyArg_UnpackTuple(args, "args", 3, 3, &row, &col, &val);
    if(!PyLong_Check(row) || !PyLong_Check(col) || (!PyFloat_Check(val) && !PyLong_Check(val))) {
        PyErr_SetString(PyExc_TypeError, "Values in args are not of correct type.");
        return NULL;
    }

    if (PyLong_AsLong(row) < 0 || PyLong_AsLong(row) >= self->mat->rows || PyLong_AsLong(col) < 0 || PyLong_AsLong(col) >= self->mat->cols) {
        PyErr_SetString(PyExc_IndexError, "i or j is out of bounds.");
        return NULL;
    }
    set(self->mat, PyLong_AsLong(row), PyLong_AsLong(col), PyFloat_AsDouble(val));
    return Py_None;
}

/*
 * Given a numc.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrix61c_get_value(Matrix61c *self, PyObject* args) {
    if(!PyTuple_Check(args) || PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "Args is not of right type or size.");
        return NULL;
    }

    PyObject *row = PyTuple_GetItem(args, 0);
    PyObject *col = PyTuple_GetItem(args, 1);
    if(!PyLong_Check(row) || !PyLong_Check(col)) {
        PyErr_SetString(PyExc_TypeError, "Values in args are not of correct type.");
        return NULL;
    }

    int i = PyLong_AsLong(row);
    int j = PyLong_AsLong(col);
    if (i < 0 || i >= self->mat->rows || j < 0 || j >= self->mat->cols) {
        PyErr_SetString(PyExc_IndexError, "i or j is out of bounds.");
        return NULL;
    }
    return PyFloat_FromDouble((get(self->mat, i, j)));
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 * Name the python function corresponding to Matrix61c_get_value as "get" and Matrix61c_set_value
 * as "set"
 * You might find this link helpful: https://docs.python.org/3.6/c-api/structures.html
 */
PyMethodDef Matrix61c_methods[] = {
    {"set", (PyCFunction) Matrix61c_set_value, METH_VARARGS, NULL},
    {"get", (PyCFunction) Matrix61c_get_value, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}   
};

/* INDEXING */

/*
 * Given a numc.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrix61c_subscript(Matrix61c* self, PyObject* key) {  
    if (self->mat->rows <=0 || self->mat->cols <=0) {
        PyErr_SetString(PyExc_ValueError, "Rows or cols is out of bounds");
        return NULL;
    }

    Matrix61c *py_mat = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);  
    int alloc_failed = allocate_matrix(&py_mat->mat, self->mat->rows, self->mat->cols);
    if (alloc_failed) return NULL;
    py_mat->shape = get_shape(self->mat->rows, self->mat->cols);
    
    
    if (PyLong_Check(key)) {
        int k = PyLong_AsLong(key);
        //return the key'th double for 1D matrix
        if (self->mat->is_1d) {
            if (self->mat->rows == 1 && k < self->mat->cols) {
                if (k < 0 || k >= self->mat->cols) {
                    PyErr_SetString(PyExc_IndexError, "Index is out of bounds");
                    return NULL;
                }
                return PyFloat_FromDouble(self->mat->data[0][k]);
            } else if (self->mat->cols == 1 && k < self->mat->rows) {
                if (k < 0 || k >= self->mat->rows) {
                    PyErr_SetString(PyExc_IndexError, "Index is out of bounds");
                    return NULL;
                }
                return PyFloat_FromDouble(self->mat->data[k][0]);
            }
        }

        //2D case for int key
        if (k < 0 || k >= self->mat->rows) {
            PyErr_SetString(PyExc_IndexError, "Index is out of bounds");
            return NULL;
        }
        //return the key'th row of 2D matrix
        int alloc_failed = allocate_matrix_ref(&py_mat->mat, self->mat, k, 0, 1, self->mat->cols);
        if(alloc_failed) return NULL;
        py_mat->shape = get_shape(1, self->mat->cols);
        return (PyObject *) py_mat;

    } else if (PySlice_Check(key)) {
        Py_ssize_t start = 0; 
        Py_ssize_t stop = 0;
        Py_ssize_t step = 0; 
        Py_ssize_t slicelength = 0;

        //Special case for 1D matrices
        if (self->mat->is_1d) {
            if (self->mat->rows == 1) {
                int valid_slice = PySlice_GetIndicesEx(key, self->mat->cols, &start, &stop, &step, &slicelength);
                if (valid_slice) {
                    PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                    return NULL;
                } else if (slicelength < 1 || step != 1) {
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                    return NULL;
                }
   
                //row matrix with length 1 -> double
                if (slicelength == 1) return PyFloat_FromDouble(self->mat->data[0][start]);
                //row matrix with length > 1 -> list
                alloc_failed = allocate_matrix_ref(&py_mat->mat, self->mat, 0, start, 1, slicelength);
                if (alloc_failed) return NULL;
                py_mat->shape = get_shape(1, slicelength);  
                return (PyObject *) py_mat;
            } else if (self->mat->cols == 1) {
                int valid_slice = PySlice_GetIndicesEx(key, self->mat->rows, &start, &stop, &step, &slicelength);
                if (valid_slice) {
                    PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                    return NULL;
                } else if (slicelength < 1 || step != 1) {
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                    return NULL;
                }

                //col matrix with length 1 -> double
                if (slicelength == 1) return PyFloat_FromDouble(self->mat->data[start][0]);
                //col matrix with length > 1 -> list
                alloc_failed = allocate_matrix_ref(&py_mat->mat, self->mat, start, 0, slicelength, 1);
                if (alloc_failed) return NULL;
                py_mat->shape = get_shape(slicelength, 1);
                return (PyObject *) py_mat; 
            }
        } else {
            int valid_slice = PySlice_GetIndicesEx(key, self->mat->rows, &start, &stop, &step, &slicelength);
            if (valid_slice) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return NULL;
            } else if (slicelength < 1 || step != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return NULL;
            }
            alloc_failed = allocate_matrix_ref(&py_mat->mat, self->mat, start, 0, slicelength, self->mat->cols);
            if (alloc_failed) return NULL;
            py_mat->shape = get_shape(slicelength, self->mat->cols);
            return (PyObject *) py_mat;
        }
       
    } else if (PyTuple_Check(key)) {
        if (self->mat->is_1d) {
            PyErr_SetString(PyExc_TypeError, "Key cannot be tuple for 1D matrix");
            return NULL;
        }
        if (PyTuple_Size(key) != 2) {
            PyErr_SetString(PyExc_TypeError, "Tuple must be of length 2");
            return NULL;
        }

        PyObject* key0 = PyTuple_GetItem(key, 0);
        PyObject* key1 = PyTuple_GetItem(key, 1);

        if (PyLong_Check(key0) && PyLong_Check(key1)) {
            int k0 = PyLong_AsLong(key0);
            int k1 = PyLong_AsLong(key1);
            if (k0 < 0 || k0 >= self->mat->rows || k1 < 0 || k1 >= self->mat->cols) {
                PyErr_SetString(PyExc_IndexError, "Key0 or Key1 is out of bounds");
                return NULL;
            }
            return PyFloat_FromDouble(self->mat->data[k0][k1]);
        }

        else if (PySlice_Check(key0) && PyLong_Check(key1)) {
            Py_ssize_t start0 = 0; 
            Py_ssize_t stop0 = 0;
            Py_ssize_t step0 = 0; 
            Py_ssize_t slicelength0 = 0;
            int valid_slice = PySlice_GetIndicesEx(key0, self->mat->cols, &start0, &stop0, &step0, &slicelength0);
            if (valid_slice) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return NULL;
            } else if (slicelength0 < 1 || step0 != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return NULL;
            }

            int k1 = PyLong_AsLong(key1);
            if (k1 < 0 || k1 >= self->mat->cols) {
                PyErr_SetString(PyExc_IndexError, "Key1 is out of bounds");
                return NULL;
            }
            
            if (slicelength0 == 1) {
                return PyFloat_FromDouble(self->mat->data[start0][k1]);
            } else {
                int alloc_failed = allocate_matrix_ref(&py_mat->mat, self->mat, start0, k1, slicelength0, 1);
                if (alloc_failed) return NULL;
                py_mat->shape = get_shape(slicelength0, 1);
                return (PyObject *) py_mat;
            }
        }

        else if (PyLong_Check(key0) && PySlice_Check(key1)) {
            Py_ssize_t start1 = 0; 
            Py_ssize_t stop1 = 0;
            Py_ssize_t step1 = 0; 
            Py_ssize_t slicelength1 = 0;
            int valid_slice = PySlice_GetIndicesEx(key1, self->mat->rows, &start1, &stop1, &step1, &slicelength1);
            if (valid_slice) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return NULL;
            } else if (slicelength1 < 1 || step1 != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return NULL;
            }

            int k0 = PyLong_AsLong(key0);
            if (k0 < 0 || k0 >= self->mat->rows) {
                PyErr_SetString(PyExc_IndexError, "Key0 is out of bounds");
                return NULL;
            }
            
            if (slicelength1 == 1) {
                return PyFloat_FromDouble(self->mat->data[k0][start1]);
            } else {
                int alloc_failed = allocate_matrix_ref(&py_mat->mat, self->mat, k0, start1, 1, slicelength1);
                if (alloc_failed) return NULL;
                py_mat->shape = get_shape(1, slicelength1);
                return (PyObject *) py_mat;
            }
        }

        else if (PySlice_Check(key0) && PySlice_Check(key1)) {
            Py_ssize_t start0 = 0;  Py_ssize_t start1 = 0; 
            Py_ssize_t stop0 = 0; Py_ssize_t stop1 = 0;
            Py_ssize_t step0 = 0; Py_ssize_t step1 = 0;
            Py_ssize_t slicelength0 = 0; Py_ssize_t slicelength1 = 0;
            int valid_slice0 = PySlice_GetIndicesEx(key0, self->mat->cols, &start0, &stop0, &step0, &slicelength0);
            if (valid_slice0) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return NULL;
            } else if (slicelength0 < 1 || step0 != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return NULL;
            }
            int valid_slice1 = PySlice_GetIndicesEx(key1, self->mat->rows, &start1, &stop1, &step1, &slicelength1);
            if (valid_slice1) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return NULL;
            } else if (slicelength1 < 1 || step1 != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return NULL;
            }

            if (slicelength0 == 1 && slicelength1 == 1) {
                return PyFloat_FromDouble(self->mat->data[start0][start1]);
            } else if (slicelength0 == 1 && slicelength1 > 1) {
                int alloc_failed = allocate_matrix_ref(&py_mat->mat, self->mat, start0, start1, 1, slicelength1);
                if (alloc_failed) return NULL;
                py_mat->shape = get_shape(1, slicelength1);
                return (PyObject *) py_mat;
            } else if (slicelength1 == 1 && slicelength0 > 1) {
                int alloc_failed = allocate_matrix_ref(&py_mat->mat, self->mat, start0, start1, slicelength0, 1);
                if (alloc_failed) return NULL;
                py_mat->shape = get_shape(slicelength0, 1);
                return (PyObject *) py_mat;
            } else if (slicelength1 > 1 && slicelength0 > 1) {
                int alloc_failed = allocate_matrix_ref(&py_mat->mat, self->mat, start0, start1, slicelength0, slicelength1);
                if (alloc_failed) return NULL;
                py_mat->shape = get_shape(slicelength0, slicelength1);
                return (PyObject *) py_mat;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Key must be an int or slice");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Key must be an int, tuple, or slice");
        return NULL;
    }
    return NULL;
}

/*
 * Given a numc.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrix61c_set_subscript(Matrix61c* self, PyObject *key, PyObject *v) {
    if (self->mat->rows <=0 || self->mat->cols <=0) {
        PyErr_SetString(PyExc_ValueError, "Rows or cols is out of bounds");
        return -1;
    }

    if (PyLong_Check(key)) {
        int k = PyLong_AsLong(key);
        //Sets the k'th element of 1D matrix
        if (self->mat->is_1d) {
            if (!PyFloat_Check(v) && !PyLong_Check(v)) {
                PyErr_SetString(PyExc_TypeError, "Value must be float or double.");
                return -1;
            }
            double val = PyFloat_AsDouble(v);
            if (self->mat->rows == 1 && k < self->mat->cols) {
                if (k < 0 || k >= self->mat->cols) {
                    PyErr_SetString(PyExc_IndexError, "Index is out of bounds");
                    return -1;
                }
                self->mat->data[0][k] = val;
                return 0;
            } else if (self->mat->cols == 1 && k < self->mat->rows) {
                if (k < 0 || k >= self->mat->rows) {
                    PyErr_SetString(PyExc_IndexError, "Index is out of bounds");
                    return -1;
                }
                self->mat->data[k][0] = val;
                return 0;
            }
        }
        
        //Check that k is in bounds
        if (k < 0 || k >= self->mat->rows) {
            PyErr_SetString(PyExc_IndexError, "Index5 is out of bounds");
            return -1;
        }
        //v must be list for 2D matrix
        if (!PyList_Check(v)) {
            PyErr_SetString(PyExc_TypeError, "Value is not a list");
            return -1;
        }
        if (PyList_Size(v) != self->mat->cols) {
            PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
            return -1;
        }
        for (int i = 0; i < self->mat->cols; i++) {
            PyObject *val = PyList_GetItem(v, i);
            if (!PyFloat_Check(val) && !PyLong_Check(val)) {
                PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                return -1;
            }
        }
        //set the k'th row of the 2d matrix
        for (int c = 0; c < self->mat->cols; c++) {
            double val = PyFloat_AsDouble(PyList_GetItem(v, c));
            self->mat->data[k][c] = val;
        }
        return 0;

    } else if (PySlice_Check(key)) {
        Py_ssize_t start = 0; 
        Py_ssize_t stop = 0; 
        Py_ssize_t step = 0; 
        Py_ssize_t slicelength = 0;

        if (self->mat->is_1d) {
            if (self->mat->rows == 1) {
                int valid_slice = PySlice_GetIndicesEx(key, self->mat->cols, &start, &stop, &step, &slicelength);
                if (valid_slice) {
                    PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                    return -1;
                } else if (slicelength < 1 || step != 1) {
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                    return -1;
                }
   
                if (slicelength == 1) {
                    //row matrix with length 1 -> double
                    if (!PyFloat_Check(v) && !PyLong_Check(v)) {
                        PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                        return -1;
                    }
                    self->mat->data[0][start] = PyFloat_AsDouble(v);
                    return 0;
                } else {
                    //row vector with slicelength > 1 (v = 1d list)
                    if (!PyList_Check(v)) {
                        PyErr_SetString(PyExc_TypeError, "Value is not a list");
                        return -1;
                    }
                    if (PyList_Size(v) != slicelength) {
                        PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                        return -1;
                    }
                    for (int i = 0; i < slicelength; i++) {
                        PyObject *val = PyList_GetItem(v, i);
                        if (!PyFloat_Check(val) && !PyLong_Check(val)) {
                            PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                            return -1;
                        }
                    }
                    
                    for (int c = start; c < stop; c++) {
                        double val = PyFloat_AsDouble(PyList_GetItem(v, c - start));
                        self->mat->data[0][c] = val;
                    }
                    return 0;
                }
            } else if (self->mat->cols == 1) {
                int valid_slice = PySlice_GetIndicesEx(key, self->mat->rows, &start, &stop, &step, &slicelength);
                if (valid_slice) {
                    PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                    return -1;
                } else if (slicelength < 1 || step != 1) {
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                    return -1;
                }

                //col matrix with length 1 -> double
                if (slicelength == 1) {
                    //col vector with slicelength = 1 (v = long)
                    if (!PyFloat_Check(v) && !PyLong_Check(v)) {
                        PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                        return -1;
                    }
                    self->mat->data[start][0] = PyFloat_AsDouble(v);
                    return 0;
                } else {
                    //col vector with slicelength > 1 (v = 1d list)
                    if (!PyList_Check(v)) {
                        PyErr_SetString(PyExc_TypeError, "Value is not a list");
                        return -1;
                    }
                    if (PyList_Size(v) != slicelength) {
                        PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                        return -1;
                    }
                    for (int i = 0; i < slicelength; i++) {
                        PyObject *val = PyList_GetItem(v, i);
                        if (!PyFloat_Check(val) && !PyLong_Check(val)) {
                            PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                            return -1;
                        }
                    }
                    for (int r = start; r < stop; r++) {
                        double val = PyFloat_AsDouble(PyList_GetItem(v, r - start));
                        self->mat->data[r][0] = val;
                    }
                    return 0;
                }
            }
        } else {
            int valid_slice = PySlice_GetIndicesEx(key, self->mat->rows, &start, &stop, &step, &slicelength);
            if (valid_slice) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return -1;
            } else if (slicelength < 1 || step != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return -1;
            }
            
            // 2D matrix case (v must be some sort of list)
            if (!PyList_Check(v)) {
                PyErr_SetString(PyExc_TypeError, "Value is not a list");
                return -1;
            }
            if (slicelength == 1) {
                if (PyList_Size(v) != self->mat->cols) {
                    PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                    return -1;
                }
                //2D matrix has slice length 1 (v = list)
                for (int i = 0; i < self->mat->cols; i++) {
                    PyObject *val = PyList_GetItem(v, i);
                    if (!PyFloat_Check(val) && !PyLong_Check(val)) {
                        PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                        return -1;
                    }
                }

                for (int c = 0; c < self->mat->cols; c++) {
                    double val = PyFloat_AsDouble(PyList_GetItem(v, c));
                    self->mat->data[start][c] = val;
                }
                return 0;
            } else {
                if (PyList_Size(v) != slicelength) {
                    PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                    return -1;
                }
                //2D matrix has slicelength > 1 (v = 2D list)
                for (int i = 0; i < PyList_Size(v); i++) {
                    PyObject* val = PyList_GetItem(v, i);
                    if (!PyList_Check(val)) {
                        PyErr_SetString(PyExc_TypeError, "Value is not a list");
                        return -1;
                    }
                    if (PyList_Size(v) != slicelength) {
                        PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                        return -1;
                    }
                    for (int j = 0; j < slicelength; j++) {
                        PyObject *value = PyList_GetItem(val, j);
                        if (!PyFloat_Check(value) && !PyLong_Check(value)) {
                            PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                            return -1;
                        }
                    }
                }
                for (int r = start; r < stop; r++) {
                    for (int c = 0; c < self->mat->cols; c++) {
                        double val = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(v, r - start), c));
                        self->mat->data[r][c] = val;
                    }
                }
                return 0;
            }
        }
    } else if (PyTuple_Check(key)) {
        if (self->mat->is_1d) {
            PyErr_SetString(PyExc_TypeError, "Key cannot be tuple for 1D matrix");
            return -1;
        }
        if (PyTuple_Size(key) != 2) {
            PyErr_SetString(PyExc_TypeError, "Tuple must be of length 2");
            return -1;
        }

        PyObject* key0 = PyTuple_GetItem(key, 0);
        PyObject* key1 = PyTuple_GetItem(key, 1);

        if (PyLong_Check(key0) && PyLong_Check(key1)) {
            int k0 = PyLong_AsLong(key0);
            int k1 = PyLong_AsLong(key1);
            if (k0 < 0 || k0 >= self->mat->rows || k1 < 0 || k1 >= self->mat->cols) {
                PyErr_SetString(PyExc_IndexError, "Index is out of bounds");
                return -1;
            }
            if (!PyFloat_Check(v) && !PyLong_Check(v)) {
                PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                return -1;
            }
            double val = PyFloat_AsDouble(v);
            self->mat->data[k0][k1] = val;
            return 0;
        } else if (PyLong_Check(key0) && PySlice_Check(key1)) {     
            Py_ssize_t start1 = 0; 
            Py_ssize_t stop1 = 0;
            Py_ssize_t step1 = 0; 
            Py_ssize_t slicelength1 = 0;
            int valid_slice = PySlice_GetIndicesEx(key1, self->mat->cols, &start1, &stop1, &step1, &slicelength1);
            if (valid_slice) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return -1;
            } else if (slicelength1 < 1 || step1 != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return -1;
            }

            int k0 = PyLong_AsLong(key0);
            if (k0 < 0 || k0 >= self->mat->rows) {
                PyErr_SetString(PyExc_IndexError, "Key0 is out of bounds");
                return -1;
            }
            
            if (slicelength1 == 1) {
                //slice lengths = 1 (v = long)
                if (!PyFloat_Check(v) && !PyLong_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                    return -1;
                }
                double val = PyFloat_AsDouble(v);
                self->mat->data[k0][start1] = val;
                return 0;
            } else {
                //slice lengths > 1 (v = 1d list)
                if (!PyList_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Value is not a list");
                    return -1;
                }   
                if (PyList_Size(v) != slicelength1) {
                    PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                    return -1;
                }
                for (int i = 0; i < slicelength1; i++) {
                    PyObject *val = PyList_GetItem(v, i);
                    if (!PyFloat_Check(val) && !PyLong_Check(val)) {
                        PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                        return -1;
                    }
                }
                for (int c = start1; c < stop1; c++) {
                    double val = PyFloat_AsDouble(PyList_GetItem(v, c - start1));
                    self->mat->data[k0][c] = val;
                }
                return 0;
            }
        } else if (PySlice_Check(key0) && PyLong_Check(key1)) {
            Py_ssize_t start0 = 0; 
            Py_ssize_t stop0 = 0;
            Py_ssize_t step0 = 0; 
            Py_ssize_t slicelength0 = 0;
            int valid_slice = PySlice_GetIndicesEx(key0, self->mat->rows, &start0, &stop0, &step0, &slicelength0);
            if (valid_slice) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return -1;
            } else if (slicelength0 < 1 || step0 != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return -1;
            }

            int k1 = PyLong_AsLong(key1);
            if (k1 < 0 || k1 >= self->mat->cols) {
                PyErr_SetString(PyExc_IndexError, "Key1 is out of bounds");
                return -1;
            }
            
            if (slicelength0 == 1) {
                //slice lengths = 1 (v = long)
                if (!PyFloat_Check(v) && !PyLong_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                    return -1;
                }
                self->mat->data[start0][k1] = PyFloat_AsDouble(v);
                return 0;
            } else {
                //slice length > 1 (v = list)
                if (!PyList_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Value is not a list");
                    return -1;
                }
                if (PyList_Size(v) != slicelength0) {
                    PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                    return -1;
                }
                for (int i = 0; i < slicelength0; i++) {
                    PyObject *val = PyList_GetItem(v, i);
                    if (!PyFloat_Check(val) && !PyLong_Check(val)) {
                        PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                        return -1;
                    }
                }
                for (int r = start0; r < stop0; r++) {
                    double val = PyFloat_AsDouble(PyList_GetItem(v, r - start0));
                    self->mat->data[r][k1] = val;
                }
                return 0;
            }
        } else if (PySlice_Check(key0) && PySlice_Check(key1)) {
            Py_ssize_t start0 = 0;  Py_ssize_t start1 = 0; 
            Py_ssize_t stop0 = 0; Py_ssize_t stop1 = 0;
            Py_ssize_t step0 = 0; Py_ssize_t step1 = 0;
            Py_ssize_t slicelength0 = 0; Py_ssize_t slicelength1 = 0;
            int valid_slice0 = PySlice_GetIndicesEx(key0, self->mat->rows, &start0, &stop0, &step0, &slicelength0);
            if (valid_slice0) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return -1;
            } else if (slicelength0 < 1 || step0 != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return -1;
            }
            int valid_slice1 = PySlice_GetIndicesEx(key1, self->mat->cols, &start1, &stop1, &step1, &slicelength1);
            if (valid_slice1) {
                PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurred");
                return -1;
            } else if (slicelength1 < 1 || step1 != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid");
                return -1;
            }
            if (slicelength0 == 1 && slicelength1 == 1) {
                //slice lengths both equal 1 (v = double)
                if (!PyFloat_Check(v) && !PyLong_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                    return -1;
                }
                double val = PyFloat_AsDouble(v);
                self->mat->data[start0][start1] = val;
                return 0;
            } else if (slicelength0 == 1 && slicelength1 > 1) {
                //slicelength0 = 1, slicelength1 > 1 (v = 1d list)
                if (!PyList_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Value is not a list");
                    return -1;
                }
                if (PyList_Size(v) != slicelength1) {
                    PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                    return -1;
                }
                for (int i = 0; i < slicelength1; i++) {
                    PyObject *val = PyList_GetItem(v, i);
                    if (!PyFloat_Check(val) && !PyLong_Check(val)) {
                        PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                        return -1;
                    }
                }
                for (int c = start1; c < stop1; c++) {
                    double val = PyFloat_AsDouble(PyList_GetItem(v, c - start1));
                    self->mat->data[start0][c] = val;
                }
                return 0;
            } else if (slicelength1 == 1 && slicelength0 > 1) {
                if (!PyList_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Value is not a list");
                    return -1;
                }
                if (PyList_Size(v) != slicelength0) {
                    PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                    return -1;
                }
                for (int i = 0; i < slicelength0; i++) {
                    PyObject *val = PyList_GetItem(v, i);
                    if (!PyFloat_Check(val) && !PyLong_Check(val)) {
                        PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                        return -1;
                    }
                }
                for (int r = start0; r < stop0; r++) {
                    double val = PyFloat_AsDouble(PyList_GetItem(v, r - start0));
                    self->mat->data[r][start1] = val;
                }
                return 0;
            } else if (slicelength1 > 1 && slicelength0 > 1) {
                if (!PyList_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Value is not a list");
                    return -1;
                }
                if (PyList_Size(v) != slicelength0) {
                    PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                    return -1;
                }
                for (int i = 0; i < PyList_Size(v); i++) {
                    PyObject* val = PyList_GetItem(v, i);
                    if (!PyList_Check(val)) {
                        PyErr_SetString(PyExc_TypeError, "Value is not a list");
                        return -1;
                    }
                    if (PyList_Size(v) != slicelength1) {
                        PyErr_SetString(PyExc_ValueError, "Value has incorrect size");
                        return -1;
                    }
                    for (int j = 0; j < slicelength1; j++) {
                        PyObject *value = PyList_GetItem(val, j);
                        if (!PyFloat_Check(value) && !PyLong_Check(value)) {
                            PyErr_SetString(PyExc_TypeError, "Value must be float or double");
                            return -1;
                        }
                    }
                }
                //Setting values for the 2D splice matrix
                for (int r = start0; r < stop0; r++) {
                    for (int c = start1; c < stop1; c++) {
                        double val = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(v, r - start0), c - start1));
                        self->mat->data[r][c] = val;
                    }
                }
                return 0;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Key must be an int or slice");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Key must be an int, slice, or tuple");
        return -1;
    }
    return -1;
}

PyMappingMethods Matrix61c_mapping = {
    NULL,
    (binaryfunc) Matrix61c_subscript,
    (objobjargproc) Matrix61c_set_subscript,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrix61c_members[] = {
    {
        "shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
        "(rows, cols)"
    },
    {NULL}  /* Sentinel */
};

PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrix61c),
    .tp_dealloc = (destructor)Matrix61c_dealloc,
    .tp_repr = (reprfunc)Matrix61c_repr,
    .tp_as_number = &Matrix61c_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrix61c_methods,
    .tp_members = Matrix61c_members,
    .tp_as_mapping = &Matrix61c_mapping,
    .tp_init = (initproc)Matrix61c_init,
    .tp_new = Matrix61c_new
};


struct PyModuleDef numcmodule = {
    PyModuleDef_HEAD_INIT,
    "numc",
    "Numc matrix operations",
    -1,
    Matrix61c_class_methods
};

/* Initialize the numc module */
PyMODINIT_FUNC PyInit_numc(void) {
    PyObject* m;

    if (PyType_Ready(&Matrix61cType) < 0)
        return NULL;

    m = PyModule_Create(&numcmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Matrix61cType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&Matrix61cType);
    printf("CS61C Fall 2020 Project 4: numc imported!\n");
    fflush(stdout);
    return m;
}
