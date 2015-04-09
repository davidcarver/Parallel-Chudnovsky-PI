/* Pi computation using Chudnovsky's algortithm.

 * Copyright 2002, 2005 Hanhong Xue (macroxue at yahoo dot com)

 * Slightly modified 2005 by Torbjorn Granlund (tege at swox dot com) to allow
   more than 2G digits to be computed.

 * Modifed 2008 by David Carver (dcarver at tacc dot utexas dot edu) to enable
   multi-threading using the algorithm from "Computation of High-Precision 
   Mathematical Constants in a Combined Cluster and Grid Environment" by 
   Daisuke Takahashi, Mitsuhisa Sato, and Taisuke Boku.  

   For gcc 4.3 or later
   gcc -Wall -O2 -fopenmp raspberry-pi2 raspberry-pi2.c -lgmp -lm

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
#include <string.h>
#include <time.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <gmp.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define A   13591409
#define B   545140134
#define C   640320
#define D   12

#define BITS_PER_DIGIT   3.32192809488736234787
#define DIGITS_PER_ITER  14.1816474627254776555
#define DOUBLE_PREC      53

char *prog_name;

////////////////////////////////////////////////////////////////////////////

double wall_clock()
{
        struct timeval timeval;
        (void) gettimeofday (&timeval, (void *) 0);
        return (double)timeval.tv_sec + 
               (double) timeval.tv_usec / 1000000.0;
}

double cpu_time()
{
        struct rusage rusage;
	(void) getrusage( RUSAGE_SELF, &rusage );
        return (double)rusage.ru_utime.tv_sec +
               (double)rusage.ru_utime.tv_usec / 1000000.0;
}

////////////////////////////////////////////////////////////////////////////

int      out=0;
mpz_t   **pstack, **qstack, **gstack;
long int threads=1, depth, cores_depth;

// binary splitting
void sum(unsigned long i, unsigned long j, unsigned long gflag)
{
  mpz_mul(pstack[i][0], pstack[i][0], pstack[j][0]);
  mpz_mul(qstack[i][0], qstack[i][0], pstack[j][0]);
  mpz_mul(qstack[j][0], qstack[j][0], gstack[i][0]);
  mpz_add(qstack[i][0], qstack[i][0], qstack[j][0]);
  if (gflag) {
     mpz_mul(gstack[i][0], gstack[i][0], gstack[j][0]);
  }
}
void bs(unsigned long a, unsigned long b, unsigned long gflag, 
        unsigned long index, unsigned long top)
{
  unsigned long mid;

  if ((b > a) && (b-a==1)) {
    /*
      g(b-1,b) = (6b-5)(2b-1)(6b-1)
      p(b-1,b) = b^3 * C^3 / 24
      q(b-1,b) = (-1)^b*g(b-1,b)*(A+Bb).
    */
    mpz_set_ui(pstack[index][top], b);
    mpz_mul_ui(pstack[index][top], pstack[index][top], b);
    mpz_mul_ui(pstack[index][top], pstack[index][top], b);
    mpz_mul_ui(pstack[index][top], pstack[index][top], (C/24)*(C/24));
    mpz_mul_ui(pstack[index][top], pstack[index][top], C*24);

    mpz_set_ui(gstack[index][top], 2*b-1);
    mpz_mul_ui(gstack[index][top], gstack[index][top], 6*b-1);
    mpz_mul_ui(gstack[index][top], gstack[index][top], 6*b-5);

    mpz_set_ui(qstack[index][top], b);
    mpz_mul_ui(qstack[index][top], qstack[index][top], B);
    mpz_add_ui(qstack[index][top], qstack[index][top], A);
    mpz_mul   (qstack[index][top], qstack[index][top], gstack[index][top]);
    if (b%2)
      mpz_neg(qstack[index][top], qstack[index][top]);

  } else {
    /*
      p(a,b) = p(a,m) * p(m,b)
      g(a,b) = g(a,m) * g(m,b)
      q(a,b) = q(a,m) * p(m,b) + q(m,b) * g(a,m)
    */
    mid = a+(b-a)*0.5224;     // tuning parameter
    bs(a, mid, 1, index, top);

    bs(mid, b, gflag, index, top+1);

    mpz_mul(pstack[index][top], pstack[index][top], pstack[index][top+1]);
    mpz_mul(qstack[index][top], qstack[index][top], pstack[index][top+1]);
    mpz_mul(qstack[index][top+1], qstack[index][top+1], gstack[index][top]);
    mpz_add(qstack[index][top], qstack[index][top], qstack[index][top+1]);
    if (gflag) {
      mpz_mul(gstack[index][top], gstack[index][top], gstack[index][top+1]);
    }
  }
}

int
main(int argc, char *argv[])
{
  mpf_t  pi, qi, ci;
  long int d=100, terms, i, j, k, cores_size;
  unsigned long psize, qsize, mid;
  double begin, mid0, mid1, mid3, mid4, end;
  double wbegin, wmid0, wmid1, wmid3, wmid4, wend;

  prog_name = argv[0];

  if (argc==1) {
    fprintf(stderr,"\nSyntax: %s <digits> <option> <threads>\n",prog_name);
    fprintf(stderr,"      <digits> digits of pi to output\n");
    fprintf(stderr,"      <option> 0 - just run (default)\n");
    fprintf(stderr,"               1 - output digits\n");
    fprintf(stderr,"               2 - debug\n");
    fprintf(stderr,"      <threads> number of threads (default 1)\n");
    exit(1);
  }
  if (argc>1)
    d = strtoul(argv[1], 0, 0);
  if (argc>2)
    out = atoi(argv[2]);
  if (argc>3)
    threads = atoi(argv[3]);

  terms = d/DIGITS_PER_ITER;
  depth = 0;
  while ((1L<<depth)<terms)
    depth++;
  depth++;

  if (threads < 1) {
        fprintf(stderr,"Number of threads reset from %ld to 1\n",threads); 
        fflush(stderr);
	threads = 1;
  }
  if ((terms > 0) && (terms < threads)) {
        fprintf(stderr,"Number of threads reset from %ld to %ld\n",threads,terms); 
        fflush(stderr);
	threads = terms;
  }
  cores_depth = 0;
  while ((1L<<cores_depth)<threads)
    cores_depth++;
  cores_size=pow(2,cores_depth);

  fprintf(stderr,"#terms=%ld, depth=%ld, threads=%ld cores=%d\n", terms, depth, threads, get_nprocs());

  begin = cpu_time();
  wbegin = wall_clock();

  /* allocate stacks */
  pstack = malloc(sizeof(mpz_t)*threads);
  qstack = malloc(sizeof(mpz_t)*threads);
  gstack = malloc(sizeof(mpz_t)*threads);
  for (j = 0; j < threads; j++) {
    pstack[j] = malloc(sizeof(mpz_t)*depth);
    qstack[j] = malloc(sizeof(mpz_t)*depth);
    gstack[j] = malloc(sizeof(mpz_t)*depth);
    for (i = 0; i < depth; i++) {
      mpz_init(pstack[j][i]);
      mpz_init(qstack[j][i]);
      mpz_init(gstack[j][i]);
    }
  }

 
  /* begin binary splitting process */
  if (terms<=0) {
    mpz_set_ui(pstack[0][0],1);
    mpz_set_ui(qstack[0][0],0);
    mpz_set_ui(gstack[0][0],1);
    for (i = 1; i < threads; i++) {
       mpz_clear(pstack[i][0]);
       mpz_clear(qstack[i][0]);
       mpz_clear(gstack[i][0]);
       free(pstack[i]);
       free(qstack[i]);
       free(gstack[i]);
    }
  } else {


    mid = terms / threads; 

#ifdef _OPENMP
#pragma omp parallel for default(shared) private(i) num_threads(threads)
#endif
    for (i = 0; i < threads; i++) {
      if (i < (threads-1))
         bs(i*mid, (i+1)*mid, cores_depth, i, 0);
      else
         bs(i*mid, terms, cores_depth, i, 0);
    }
    for (j = 0; j < threads; j++) {
      for (i=1; i<depth; i++) {
        mpz_clear(pstack[j][i]);
        mpz_clear(qstack[j][i]);
        mpz_clear(gstack[j][i]);
      }
    }

    mid0 = cpu_time();
    wmid0 = wall_clock();
    fprintf(stderr,"bs1      cputime = %6.2f  wallclock = %6.2f   factor = %6.1f\n",
      mid0-begin,wmid0-wbegin,(mid0-begin)/(wmid0-wbegin));

    for (k = 1; k < cores_size; k*=2) {
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(i) num_threads(threads)
#endif
      for (i = 0; i < threads; i=i+2*k) {
        if (i+k < threads) {
          sum( i, i+k, 1);
          mpz_clear(pstack[i+k][0]);
          mpz_clear(qstack[i+k][0]);
          mpz_clear(gstack[i+k][0]);
          free(pstack[i+k]);
          free(qstack[i+k]);
          free(gstack[i+k]);
        }
      }
    }

    mid1 = cpu_time();
    wmid1 = wall_clock();
    fprintf(stderr,"bs2      cputime = %6.2f  wallclock = %6.2f   factor = %6.1f\n",
      mid1-mid0,wmid1-wmid0,(mid1-mid0)/(wmid1-wmid0));

  }
  mpz_clear(gstack[0][0]);
  free(gstack[0]);
  free(gstack);

  /* prepare to convert integers to floats */
  mpf_set_default_prec((long int)(d*BITS_PER_DIGIT+16));

  /*
	  p*(C/D)*sqrt(C)
    pi = -----------------
	     (q+A*p)
  */

  psize = mpz_sizeinbase(pstack[0][0],10);
  qsize = mpz_sizeinbase(qstack[0][0],10);

  mpz_addmul_ui(qstack[0][0], pstack[0][0], A);
  mpz_mul_ui(pstack[0][0], pstack[0][0], C/D);

  mpf_init(pi);
  mpf_set_z(pi, pstack[0][0]);
  mpz_clear(pstack[0][0]);

  mpf_init(qi);
  mpf_set_z(qi, qstack[0][0]);
  mpz_clear(qstack[0][0]);

  free(pstack[0]);
  free(qstack[0]);
  free(pstack);
  free(qstack);

  /* final step */

  mid3 = cpu_time();
  wmid3 = wall_clock();

#ifdef _OPENMP
  #pragma omp parallel sections shared(qi,pi,ci) num_threads(threads)
  {
    #pragma omp section
    {
      mpf_div(qi, pi, qi);
      mpf_clear(pi);
    }

    #pragma omp section
    {
      mpf_init(ci);
      mpf_sqrt_ui(ci, C);
    }
  }
#else
      mpf_div(qi, pi, qi);
      mpf_clear(pi);
      mpf_init(ci);
      mpf_sqrt_ui(ci, C);
#endif

  mid4 = cpu_time();
  wmid4 = wall_clock();
  fprintf(stderr,"div/sqrt cputime = %6.2f  wallclock = %6.2f   factor = %6.1f\n",
    mid4-mid3,wmid4-wmid3,(mid4-mid3)/(wmid4-wmid3));

  mpf_mul(qi, qi, ci);
  mpf_clear(ci);

  end = cpu_time();
  wend = wall_clock();
  fprintf(stderr,"mul      cputime = %6.2f  wallclock = %6.2f   factor = %6.1f\n",
    end-mid4,wend-wmid4,(end-mid4)/(wend-wmid4));

  fprintf(stderr,"total    cputime = %6.2f  wallclock = %6.2f   factor = %6.1f\n",
    end-begin,wend-wbegin,(end-begin)/(wend-wbegin));
  fflush(stderr);

  fprintf(stderr,"   P size=%ld digits (%f)\n"
	 "   Q size=%ld digits (%f)\n",
	 psize, (double)psize/d, qsize, (double)qsize/d);

  /* output Pi and timing statistics */
  if (out&1)  {
    fprintf(stdout,"pi(0,%ld)=\n", terms);
    mpf_out_str(stdout, 10, d, qi);
    fprintf(stdout,"\n");
  }

  /* free float resources */
  mpf_clear(qi);

  exit (0);
}
