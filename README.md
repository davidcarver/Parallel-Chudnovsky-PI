Parallel Chudnovsky PI program written in C using OpenMP 3.0

Repository of Parallel Chudnovsky PI programs written in C using OpenMP 3.0
 
Warning: the number of digits computed is limited by the amount of memory up to 41 billion digits.

Files
  
  * pgmp-chudnovsky-factoring-task.c    Using OpenMP "for loop" pragma with Hanhong Xue's factorization optimization
  * pgmp-chudnovsky-factoring-forloop.c Using OpenMP "for loop" pragma with Hanhong Xue's factorization optimization
  * pgmp-chudnovsky-2008-October.c      My first version parallel version  submitted to gmplib.org
                                        https://gmplib.org/list-archives/gmp-discuss/2008-October/003419.html)
  * pgmp-chudnovsky-2008-November.c     My second version With Hanhong Xue's factorization optimization using OpenMP "
                                        for loop" https://gmplib.org/list-archives/gmp-discuss/2008-November/003444.html)
  
  * gmp-chudnovsky.c                    The Hanhong Xue orginal version https://gmplib.org/download/misc/gmp-chudnovsky.c
 

Build (gcc 4.3 or later)

 * To compile using gcc with OpenMP 3.0

   gcc -Wall -fopenmp -O3 -o pgmp-chudnovsky pgmp-chudnovsky.c -lgmp -lm

Acknowledgements
 * Hanhong Xue  at https://gmplib.org/download/misc/gmp-chudnovsky.c
 * "Fast multiprecision evaluation of series of rational numbers" by Bruno Haible and Thomas Papanikolaou
 * "Computation of High-Precision Mathematical Constants in a Combined Cluster and Grid Environment" by 
   Daisuke Takahashi, Mitsuhisa Sato, and Taisuke Boku.
 * "10 Trillion Digits of Pi: A Case Study of summing Hypergeometric Series to high
   precision on Multicore Systems" by Yee and Kondo at 
   https://www.ideals.illinois.edu/items/28571/bitstreams/96266/data.pdf.
 * Mario Roy at https://github.com/marioroy/Chudnovsky-Pi

Wish list for GMP
 * Susumu Tsukamoto at  https://gmplib.org/list-archives/gmp-discuss/2021-April/006651.html


