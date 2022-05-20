This project is a simplified version of numpy, implemented using the C programming language.

Here is a summary of the speedup techniques used ot implement matrix operations:

For the more simple methods in matrix.c we utilized data level parallelism, using Intel’s intrinsics SIMD instructions to vectorize our data and improve performance of the more simple matrix.c methods, like add, sub, neg, and absolute. We also speed up our code by getting rid of function calls and code that we determined to be redundant computation.

The next step for our simple functions was to add data level parallelism through the use of the very cool library openMp. Using this library we were able to utilize the multicore design of modern day computers, while keeping in mind the possibilities of data race and other complications that may arise with the use of pragma for. 

For matrix multiplication, We unrolled some of the inner loops, used the ikj order of for loops, and incorporated thread level parallelism to improve our speed.
For lower dimensions our thread level parallelism was actually a hindrance to our performance, so we decided to have a thread count that depends on the size of the given matrices.

For the matrix power operation, we used a bit wise implementation of least squaring which we found really increased the performance to the point where other optimizations, like thread and data level parallelism, seemed unnecessary. 

