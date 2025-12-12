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
$ ./nob
```
from the root directory.

**Windows**

The already-provided binary is an ELF64 executable and won't work on windows. You'll need to first compile nob into a PAC executable
```shell
gcc -o nob.exe nob.c
./nob
```

## General information

This project contains a basic GPU linear algebra library called `cuLA` (**CU**DA **L**inear **A**lgebra). The general structure is very simple. The `src/cuLA` folder contains cuda files each outlining some routine used for calculations in linear algebra