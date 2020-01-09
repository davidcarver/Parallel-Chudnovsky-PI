Parallel Chudnovsky PI programs for Raspberry PI 2/3/4

Parallel Chudnovsky PI programs updated from orginal post at 
 https://gmplib.org/list-archives/gmp-discuss/2008-October/003418.html for Raspberry PI 2's  arm7vl quad core processor.
 
Warning: the number of digits computed is limited by the amount of memory.

Files

  * raspberry-pi2-openmp.c
  * raspberry-pi2-openmp-task.c  (modified to use OpenMP 3.0's task pragma)
  * raspberry-pi2-openmp-forloop.c  (modified to use OpenMP for loop pragma)
  * raspberry-pi2-cilk.c
  * raspberry-pi2-cilk-task.c    (modified to use Cilkplus cilk_spawn pragma)

Build (gcc 4.3 or later)

 * To compile using gcc with OpenMP
   gcc -fopenmp -o raspberry-pi2-openmp raspberry-pi2-openmp.c -lgmp -lm

 * To compile using gcc with Cilkplus 
   gcc -fcilkplus -Wall -O2 -o raspberry-pi2-cilk raspberry-pi2-cilk.c -lgmp -lm



