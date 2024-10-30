# kokkos_tests

In this repository, you will find the source codes of simple programs that I used to test and learn how to use the Kokkos library.

Tested on "home-lab" Dell R7610 machine:
* Intel(R) Xeon(R) CPU E5-2630 v2 @ 2.60GHz (2 Sockets)
* RAM Samsung DDR3 1600MHz PC3-12800R ECC (128GB)
* GPU: GeForce RTX 3060 (12GB) / Cuda driver 12.7
* OS: Proxmox 8.1.14 / Ubuntu 23.04 (Kernel: 6.5.13-1-pve)

OMP settings:\
`OMP_PROC_BIND=spread`\
`OMP_PLACE=threads`\
`OMP_NUM_THREADS=24`

More various examples and specific results may appear in the future.
