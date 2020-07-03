# Distributed Random Forest

This project's goal is to implement a distributed Random Forest.

## Getting started

### Prerequisites

In order to launch this project, you'll need:

* CMake version 3.9 or higher
* g++ version 8.1 or higher
* A working version of `open-mpi`. The project was developped using Open-MPI version 3.1.3

### Building the project
In order to build the project, type the following commands at the root of the project:
```
sh$ mkdir build
sh$ cd build
sh$ cmake ..
sh$ make
```

This will build two binaries and a library:
- lib_diraf: the library to use our Distributed Random Forest
- distributed_rf: the main binary
- test_diraf: the test binary

### Lauching the project
In order to launch the project, you can type the following command, in the `build` directory:
```
sh$ mpirun ./distributed_rf
```

or if you want multiple processes (for example 4):

```
sh$ mpirun -n 4 ./distributed_rf
```

### Lauching the test binary
In order to launch the test binary, you can type the following command, in the `build` directory:
```
sh$ mpirun ./test_diraf
```

## Authors

* **Mickael IDE** - [lowener](https://github.com/lowener)
* **Nicolas LUGASSY** - [Ringokilol](https://github.com/Ringokilol)
* **Arthur NAEGELY** - [Arthur-NA](https://github.com/Arthur-NA)

## License
This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details
