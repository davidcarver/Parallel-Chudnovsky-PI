/* Pi computation using Chudnovsky's algortithm.

 * Copyright 2002, 2005 Hanhong Xue (macroxue at yahoo dot com)

 * Slightly modified 2005 by Torbjorn Granlund (tege at swox dot com) to allow
   more than 2G digits to be computed.

 * Modifed 2008, 2020 by David Carver (dcarver at tacc dot utexas dot edu) to enable
   multi-threading using the algorithm from "Computation of High-Precision 
   Mathematical Constants in a Combined Cluster and Grid Environment" by 
   Daisuke Takahashi, Mitsuhisa Sato, and Taisuke Boku.  

 * Updated to simpilfy OpenMP and improve performance; and incorperate some excellent
   ideas Mario Roy implementation at https://github.com/marioroy/Chudnovsky-Pi. 

   To compile:
   gcc -Wall -fopenmp -O2 -o pgmp-chudnovsky pgmp-chudnovsky.c -lgmp -lm

   To run:
   ./pgmp-chudnovsky 1000 1

   To get help run the program with no options:
   ./pgmp-chudnovsky

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
 * EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/sysinfo.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <gmp.h>
#include <omp.h>

#define A   13591409
#define B   545140134
#define C   640320
#define D   12

#define BITS_PER_DIGIT   3.32192809488736234787
#define DIGITS_PER_ITER  14.1816474627254776555
#define DOUBLE_PREC      53

char *prog_name;
long d=100, out=0, cutoff=1000, threads=1;

////////////////////////////////////////////////////////////////////////////

/* https://blog.habets.se/2010/09/gettimeofday-should-never-be-used-to-measure-time.html */

double wall_clock()
{
  struct timespec timeval;

  (void) clock_gettime (CLOCK_MONOTONIC, &timeval);
  return (double) timeval.tv_sec +
         (double) timeval.tv_nsec / 1000000000.0;
}

double cpu_time()
{
  struct rusage rusage;

  (void) getrusage( RUSAGE_SELF, &rusage );
  return (double)rusage.ru_utime.tv_sec +
         (double)rusage.ru_utime.tv_usec / 1000000.0;
}

////////////////////////////////////////////////////////////////////////////

/* binary splitting */

void sum(mpz_t pstack1, mpz_t qstack1, mpz_t gstack1,
     mpz_t pstack2, mpz_t qstack2, mpz_t gstack2, long gflag)
{
   #pragma omp parallel num_threads(3)
   {
     #pragma omp single nowait
     {
       #pragma omp task shared(pstack1, pstack2)
         mpz_mul(pstack1, pstack1, pstack2);

       #pragma omp task shared(qstack1, pstack2)
         mpz_mul(qstack1, qstack1, pstack2);

       #pragma omp task shared(qstack2, gstack1)
         mpz_mul(qstack2, qstack2, gstack1);

       #pragma omp taskwait 
     }
   }

   mpz_add(qstack1, qstack1, qstack2);

   if (gflag)
     mpz_mul(gstack1, gstack1, gstack2);
}

void bs(unsigned long a, unsigned long b, unsigned long gflag, unsigned long level, 
     mpz_t pstack1, mpz_t qstack1, mpz_t gstack1)
{
  unsigned long mid;

  if (out&2)
  {
    fprintf(stderr, "bs: a = %ld b = %ld gflag = %ld level = %ld\n", a, b, gflag, level);
    fflush(stderr);
  }

  if ((b > a) && ((b-a) == 1))
  {

    /*
      g(b-1,b) = (6b-5)(2b-1)(6b-1)
      p(b-1,b) = b^3 * C^3 / 24
      q(b-1,b) = (-1)^b*g(b-1,b)*(A+Bb).
    */

    mpz_set_ui(pstack1, b);
    mpz_mul_ui(pstack1, pstack1, b);
    mpz_mul_ui(pstack1, pstack1, b);
    mpz_mul_ui(pstack1, pstack1, (C/24)*(C/24));
    mpz_mul_ui(pstack1, pstack1, C*24);

    mpz_set_ui(gstack1, 2*b-1);
    mpz_mul_ui(gstack1, gstack1, 6*b-1);
    mpz_mul_ui(gstack1, gstack1, 6*b-5);

    mpz_set_ui(qstack1, b);
    mpz_mul_ui(qstack1, qstack1, B);
    mpz_add_ui(qstack1, qstack1, A);
    mpz_mul(qstack1, qstack1, gstack1);
    if (b%2)
      mpz_neg(qstack1, qstack1);

  } else {

    mpz_t pstack2, qstack2, gstack2;

    mpz_init(pstack2);
    mpz_init(qstack2);
    mpz_init(gstack2);

    /*
      p(a,b) = p(a,m) * p(m,b)
      g(a,b) = g(a,m) * g(m,b)
      q(a,b) = q(a,m) * p(m,b) + q(m,b) * g(a,m)
    */

    mid = a+(b-a)*0.5224;     // tuning parameter

    bs(a, mid, 1, level+1, pstack1, qstack1, gstack1);
    bs(mid, b, gflag, level+1, pstack2, qstack2, gstack2);

    mpz_mul(pstack1, pstack1, pstack2);
    mpz_mul(qstack1, qstack1, pstack2);
    mpz_mul(qstack2, qstack2, gstack1);
    mpz_add(qstack1, qstack1, qstack2);

    if (gflag)
    {
      mpz_mul(gstack1, gstack1, gstack2);
    }

    mpz_clear(pstack2);
    mpz_clear(qstack2);
    mpz_clear(gstack2);
  }
}

int
main(int argc, char *argv[])
{
  mpz_t *pstack, *qstack, *gstack;
  mpf_t  pi, qi, ci;
  long terms, i, k, cores_size, cores;
  long depth, cores_depth, psize, qsize, mid;
  double begin, mid0, mid1, end;
  double wbegin, wmid0, wmid1, wend;

  prog_name = argv[0];

  if (argc==1)
  {
    fprintf(stderr, "\nSyntax: %s <digits> <option> <threads>\n", prog_name);
    fprintf(stderr, "      <digits> digits of pi to output\n");
    fprintf(stderr, "      <option> 0 - just run (default)\n");
    fprintf(stderr, "               1 - output decimal digits to stdout\n");
    fprintf(stderr, "               2 - debug\n");
    fprintf(stderr, "      <threads> number of threads (default 1)\n");
    fprintf(stderr, "      <cutoff> cutoff for recursion (default 1000)\n");
    exit(1);
  }
  if (argc>1)
    d = strtoul(argv[1], 0, 0);
  if (argc>2)
    out = atoi(argv[2]);
  if (argc>3)
    threads = atoi(argv[3]);
  if (argc>4)
    cutoff = atoi(argv[4]);

  cores = omp_get_num_procs();
  omp_set_nested(1);
  omp_set_dynamic(0);

  terms = d/DIGITS_PER_ITER;
  depth = 0;
  while ((1L<<depth)<terms)
    depth++;
  depth++;

  if (threads < 1)
  {
        fprintf(stderr, "Number of threads reset from %ld to 1\n", threads); 
        fflush(stderr);
	threads = 1;
  }
  if ((terms > 0) && (terms < threads))
  {
        fprintf(stderr, "Number of threads reset from %ld to %ld\n", threads, terms); 
        fflush(stderr);
	threads = terms;
  }

  cores_depth = 0;
  while ((1L<<cores_depth)<threads)
    cores_depth++;
  cores_size=pow(2, cores_depth);

  omp_set_num_threads(threads);

  fprintf(stderr, "#terms=%ld, depth=%ld, threads=%ld cores=%ld cutoff=%ld\n", terms, depth, threads, cores, cutoff);

  mid0 = begin = cpu_time();
  wmid0 = wbegin = wall_clock();

  /* allocate stacks */

  pstack = malloc(sizeof(mpz_t)*threads);
  qstack = malloc(sizeof(mpz_t)*threads);
  gstack = malloc(sizeof(mpz_t)*threads);
  for (i = 0; i < threads; i++)
  {
    mpz_init(pstack[i]);
    mpz_init(qstack[i]);
    mpz_init(gstack[i]);
  }

  /* begin binary splitting process */

  if (terms<=0)
  {
    mpz_set_ui(pstack[0], 1);
    mpz_set_ui(qstack[0], 0);
    mpz_set_ui(gstack[0], 1);
    for (i = 1; i < threads; i++)
    {
      mpz_clear(pstack[i]);
      mpz_clear(qstack[i]);
      mpz_clear(gstack[i]);
    }
  } else {

    mid0 = cpu_time();
    wmid0 = wall_clock();

    mid = terms / threads; 

#pragma omp parallel for default(shared) private(i) num_threads(threads)
    for (i = 0; i < threads; i++)
    {
      if (i < (threads-1))
         bs(i*mid, (i+1)*mid, 1, 0, pstack[i], qstack[i], gstack[i]);
      else
         bs(i*mid, terms, 1, 0, pstack[i], qstack[i], gstack[i]);
    }

    mid1 = cpu_time();
    wmid1 = wall_clock();
    fprintf(stderr, "bs         cputime = %6.2f  wallclock = %6.2f   factor = %6.2f\n",
      (mid1-mid0), (wmid1-wmid0), (mid1-mid0)/(wmid1-wmid0));
    fflush(stderr);

    mid0 = cpu_time();
    wmid0 = wall_clock();

    #pragma omp parallel private(k, i) num_threads(threads/2)
    {
      for (k = 1; k < cores_size; k*=2) 
      {
        #pragma omp for schedule(static,1)
        for (i = 0; i < threads; i=i+2*k)
        {
          if (i+k < threads)
          {
            long gflag = (i+2*k < threads) ? 1 : 0;
            sum(pstack[i], qstack[i], gstack[i], pstack[i+k], qstack[i+k], gstack[i+k], gflag);
            mpz_clear(pstack[i+k]);
            mpz_clear(qstack[i+k]);
            mpz_clear(gstack[i+k]);
          }
        }
        #pragma omp barrier
      }
    }
    mpz_clear(gstack[0]);
  }

  mid1 = cpu_time();
  wmid1 = wall_clock();
  fprintf(stderr, "sum        cputime = %6.2f  wallclock = %6.2f   factor = %6.2f\n",
    (mid1-mid0), (wmid1-wmid0), (mid1-mid0)/(wmid1-wmid0));
  fflush(stderr);

  /* prepare to convert integers to floats */

  mpf_set_default_prec((long)(d*BITS_PER_DIGIT+16));

  /*
	  p*(C/D)*sqrt(C)
    pi = -----------------
	     (q+A*p)
  */

  psize = mpz_sizeinbase(pstack[0], 10);
  qsize = mpz_sizeinbase(qstack[0], 10);

  mpz_addmul_ui(qstack[0], pstack[0], A);
  mpz_mul_ui(pstack[0], pstack[0], C/D);

  mpf_init(pi);
  mpf_set_z(pi, pstack[0]);
  mpz_clear(pstack[0]);

  mpf_init(qi);
  mpf_set_z(qi, qstack[0]);
  mpz_clear(qstack[0]);

  /* final step */

  mid0 = cpu_time();
  wmid0 = wall_clock();

  mpf_init(ci);
  if (threads < 2 )
  {
    mpf_div(qi, pi, qi);
    mpf_sqrt_ui(ci, C);
  } else {
    #pragma omp parallel num_threads(2)
    {
      #pragma omp single nowait
      {
        #pragma omp task shared(pi, qi)
          mpf_div(qi, pi, qi);

        #pragma omp task shared(ci)
          mpf_sqrt_ui(ci, C);

        #pragma omp taskwait 
      }
    }
  }

  mid1 = cpu_time();
  wmid1 = wall_clock();
  fprintf(stderr, "div/sqrt   cputime = %6.2f  wallclock = %6.2f   factor = %6.2f\n",
    (mid1-mid0), (wmid1-wmid0), (mid1-mid0)/(wmid1-wmid0));
  fflush(stderr);

  mid0 = cpu_time();
  wmid0 = wall_clock();

  mpf_mul(pi, qi, ci);
  mpf_clear(ci);
  mpf_clear(qi);

  mid1 = end = cpu_time();
  wmid1 = wend = wall_clock();
  fprintf(stderr, "mul        cputime = %6.2f  wallclock = %6.2f   factor = %6.2f\n",
    (mid1-mid0), (wmid1-wmid0), (mid1-mid0)/(wmid1-wmid0));
  fflush(stderr);

  fprintf(stderr, "total      cputime = %6.2f  wallclock = %6.2f   factor = %6.2f\n",
    (end-begin), (wend-wbegin), (end-begin)/(wend-wbegin));
  fflush(stderr);

  /* output Pi and timing statistics */

  fprintf(stderr, "   P size=%ld digits (%f)\n"
	 "   Q size=%ld digits (%f)\n",
	 psize, (double)psize/d, qsize, (double)qsize/d);
  fflush(stderr);

  if (out&1)
  {
    fprintf(stdout, "pi(0,%ld)=\n", terms);
    mpf_out_str(stdout, 10, d, pi);
    fprintf(stdout, "\n");
    fflush(stdout);
  }

  /* free float resources */

  mpf_clear(pi);

  exit (0);
}
