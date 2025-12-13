# Deep Learning Project

This is a personal project of mine where I want to learn the specifics of how deep learning and neural networks work. This repo is my notes and code for this purpose.

## Structure

The project structure looks something like this:
- `data/`: Training and test data for the models. (Not tracked for the moment)
- `notes/`: LaTeX notes for the mathematical concepts needed.
- `src/`: Where the project's source code lives.
	- `thirdparty/`: Where third-party libraries are located.
- `build/`: Built binaries.

## Build instructions

**Linux**

To build the project, simply run
```bash
./nob
```
from the root directory.

**Windows**

The already-provided binary is an ELF64 executable and won't work on windows. You'll need to first compile nob into a PAC executable
```shell
gcc -o nob.exe nob.c
./nob
```

## General information

This project contains a basic GPU linear algebra library called `cuLA` (**CU**DA **L**inear **A**lgebra). This "library," if we can even call it that, is essentially just a wrapper around cuBLAS functions which overloads operators in newly defined `Matrix` and `Vector` structures, allowing for code that looks like this:
```cpp
Matrix W(10,12);
Vector a(12);
Vector b(10);

// ... Setting values of W, a, and b ...

Vector z = W * a + b;
``` 

Instead of this:
```cpp
// rows
int m = 10;
// columns
int n = 12;

float* W = (float*)malloc(m * n * sizeof(float));
float* a = (float*)malloc(n * sizeof(float));
float* b = (float*)malloc(m * sizeof(float));
float* z = (float*)malloc(m * sizeof(float));

// ... Setting values of W, a, and b ...

int size_W = m * n * sizeof(float);
int size_a = n * sizeof(float);
int size_b = m * sizeof(float);
int size_z = m * sizeof(float);

float* dev_W = nullptr;
float* dev_a = nullptr;
float* dev_b = nullptr;
float* dev_z = nullptr;

cudaMalloc(&dev_W, size_W);
cudaMalloc(&dev_a, size_a);
cudaMalloc(&dev_b, size_b);
cudaMalloc(&dev_z, size_z);

// W
cublasSetMatrix(
	m, n, sizeof(float),
	W, m,
	dev_W, m
);

// a
cublasSetVector(
	n, sizeof(float),
	a, 1,
	dev_a, 1
);

// b
cublasSetVector(
	m, sizeof(float),
	b, 1,
	dev_b, 1
);

// copy b to z since z is the output
cublasSetVector(
	m, sizeof(float),
	b, 1,
	dev_z, 1
);

float alpha = 1.0f;
float beta = 1.0f;

// y = alpha * op(A) * x + beta * y
cublasSgemv(
	ctx, CUBLAS_OP_N,
	m, n,
	&alpha,
	dev_W, m,
	dev_a, 1,
	&beta,
	dev_z, 1
);
``` 