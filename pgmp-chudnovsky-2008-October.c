/* Pi computation using Chudnovsky's algortithm.

 * Copyright 2002, 2005 Hanhong Xue (macroxue at yahoo dot com)

 * Slightly modified 2005 by Torbjorn Granlund (tege at swox dot com) to allow
   more than 2G digits to be computed.

 * Modifed 2008 by David Carver (dcarver at tacc dot utexas dot edu) to enable
   multi-threading using the algorithm from "Computation of High-Precision 
   Mathematical Constants in a Combined Cluster and Grid Environment" by 
   Daisuke Takahashi, Mitsuhisa Sato, and Taisuke Boku.  

   For gcc 4.3
   gcc -fopenmp -Wall -O2 -o pgmp-chudnovsky pchudnovsky.c -lgmp -lm

   For Intel 10.1 compiler
   icc -openmp  -O2 -o pgmp-chudnovsky pgmp-chudnovsky.c -lgmp -lm

   For AIX xlc
   xlc_r -qsmp=omp -O2 -o pgmp-chudnovsky pchudnovsky.c  -lgmp -lm

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
#include "gmp.h"

#define A   13591409
#define B   545140134
#define C   640320
#define D   12

#define BITS_PER_DIGIT   3.32192809488736234787
#define DIGITS_PER_ITER  14.1816474627254776555
#define DOUBLE_PREC      53


char *prog_name;

#if CHECK_MEMUSAGE
#undef CHECK_MEMUSAGE
#define CHECK_MEMUSAGE							\
  do {									\
    char buf[100];							\
    snprintf (buf, 100,							\
	      "ps aguxw | grep '[%c]%s'", prog_name[0], prog_name+1);	\
    system (buf);							\
  } while (0)
#else
#undef CHECK_MEMUSAGE
#define CHECK_MEMUSAGE
#endif


////////////////////////////////////////////////////////////////////////////

mpf_t t1, t2;

// r = sqrt(x)
void
my_sqrt_ui(mpf_t r, unsigned long x)
{
  unsigned long prec, bits, prec0;

  prec0 = mpf_get_prec(r);

  if (prec0<=DOUBLE_PREC) {
    mpf_set_d(r, sqrt(x));
    return;
  }

  bits = 0;
  for (prec=prec0; prec>DOUBLE_PREC;) {
    int bit = prec&1;
    prec = (prec+bit)/2;
    bits = bits*2+bit;
  }

  mpf_set_prec_raw(t1, DOUBLE_PREC);
  mpf_set_d(t1, 1/sqrt(x));

  while (prec<prec0) {
    prec *=2;
    if (prec<prec0) {
      /* t1 = t1+t1*(1-x*t1*t1)/2; */
      mpf_set_prec_raw(t2, prec);
      mpf_mul(t2, t1, t1);         // half x half -> full
      mpf_mul_ui(t2, t2, x);
      mpf_ui_sub(t2, 1, t2);
      mpf_set_prec_raw(t2, prec/2);
      mpf_div_2exp(t2, t2, 1);
      mpf_mul(t2, t2, t1);         // half x half -> half
      mpf_set_prec_raw(t1, prec);
      mpf_add(t1, t1, t2);
    } else {
      prec = prec0;
      /* t2=x*t1, t1 = t2+t1*(x-t2*t2)/2; */
      mpf_set_prec_raw(t2, prec/2);
      mpf_mul_ui(t2, t1, x);
      mpf_mul(r, t2, t2);          // half x half -> full
      mpf_ui_sub(r, x, r);
      mpf_mul(t1, t1, r);          // half x half -> half
      mpf_div_2exp(t1, t1, 1);
      mpf_add(r, t1, t2);
      break;
    }
    prec -= (bits&1);
    bits /=2;
  }
}

// r = y/x   WARNING: r cannot be the same as y.
void
my_div(mpf_t r, mpf_t y, mpf_t x)
{
  unsigned long prec, bits, prec0;

  prec0 = mpf_get_prec(r);

  if (prec0<=DOUBLE_PREC) {
    mpf_set_d(r, mpf_get_d(y)/mpf_get_d(x));
    return;
  }

  bits = 0;
  for (prec=prec0; prec>DOUBLE_PREC;) {
    int bit = prec&1;
    prec = (prec+bit)/2;
    bits = bits*2+bit;
  }

  mpf_set_prec_raw(t1, DOUBLE_PREC);
  mpf_ui_div(t1, 1, x);

  while (prec<prec0) {
    prec *=2;
    if (prec<prec0) {
      /* t1 = t1+t1*(1-x*t1); */
      mpf_set_prec_raw(t2, prec);
      mpf_mul(t2, x, t1);          // full x half -> full
      mpf_ui_sub(t2, 1, t2);
      mpf_set_prec_raw(t2, prec/2);
      mpf_mul(t2, t2, t1);         // half x half -> half
      mpf_set_prec_raw(t1, prec);
      mpf_add(t1, t1, t2);
    } else {
      prec = prec0;
      /* t2=y*t1, t1 = t2+t1*(y-x*t2); */
      mpf_set_prec_raw(t2, prec/2);
      mpf_mul(t2, t1, y);          // half x half -> half
      mpf_mul(r, x, t2);           // full x half -> full
      mpf_sub(r, y, r);
      mpf_mul(t1, t1, r);          // half x half -> half
      mpf_add(r, t1, t2);
      break;
    }
    prec -= (bits&1);
    bits /=2;
  }
}

////////////////////////////////////////////////////////////////////////////

int      out=0;
mpz_t   **pstack, **qstack, **gstack;
long int cores=1, depth, cores_depth;
double   progress=0, percent;

// binary splitting
void
sum(unsigned long i, unsigned long j, unsigned long gflag)
{
  mpz_mul(pstack[i][0], pstack[i][0], pstack[j][0]);
  mpz_mul(qstack[i][0], qstack[i][0], pstack[j][0]);
  mpz_mul(qstack[j][0], qstack[j][0], gstack[i][0]);
  mpz_add(qstack[i][0], qstack[i][0], qstack[j][0]);
  if (gflag) {
     mpz_mul(gstack[i][0], gstack[i][0], gstack[j][0]);
  }
}
void
bs(unsigned long a, unsigned long b, unsigned long gflag, unsigned long level, unsigned long index, unsigned long top)
{
  unsigned long mid;
  int ccc;

  if (out&2) {
    fprintf(stderr,"bs: a = %ld b = %ld gflag = %ld index = %ld level = %ld top = %ld \n", a,b,gflag,index,level,top);
    fflush(stderr);
  }

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

    if (b>(int)(progress)) {
      fprintf(stderr,"."); fflush(stderr);
      progress += percent*2;
    }

  } else {
    /*
      p(a,b) = p(a,m) * p(m,b)
      g(a,b) = g(a,m) * g(m,b)
      q(a,b) = q(a,m) * p(m,b) + q(m,b) * g(a,m)
    */
    mid = a+(b-a)*0.5224;     // tuning parameter
    bs(a, mid, 1, level+1, index, top);

    bs(mid, b, gflag, level+1, index, top+1);

    ccc = level == 0;

    if (ccc) CHECK_MEMUSAGE;
    mpz_mul(pstack[index][top], pstack[index][top], pstack[index][top+1]);

    if (ccc) CHECK_MEMUSAGE;
    mpz_mul(qstack[index][top], qstack[index][top], pstack[index][top+1]);

    if (ccc) CHECK_MEMUSAGE;
    mpz_mul(qstack[index][top+1], qstack[index][top+1], gstack[index][top]);

    if (ccc) CHECK_MEMUSAGE;
    mpz_add(qstack[index][top], qstack[index][top], qstack[index][top+1]);

    if (gflag) {
      mpz_mul(gstack[index][top], gstack[index][top], gstack[index][top+1]);
    }
  }
}

int
main(int argc, char *argv[])
{
  mpf_t  pi, qi;
  long int d=100, terms, i, j, k, cores_size;
  unsigned long psize, qsize, mid;
  clock_t begin, mid0, mid1, mid2, mid3, mid4, end;

  prog_name = argv[0];

  if (argc==1) {
    fprintf(stderr,"\nSyntax: %s <digits> <option> <cores>\n",prog_name);
    fprintf(stderr,"      <digits> digits of pi to output\n");
    fprintf(stderr,"      <option> 0 - just run (default)\n");
    fprintf(stderr,"               1 - output digits\n");
    fprintf(stderr,"               2 - debug\n");
    fprintf(stderr,"      <cores> number of cores (default 1)\n");
    exit(1);
  }
  if (argc>1)
    d = strtoul(argv[1], 0, 0);
  if (argc>2)
    out = atoi(argv[2]);
  if (argc>3)
    cores = atoi(argv[3]);

  terms = d/DIGITS_PER_ITER;
  depth = 0;
  while ((1L<<depth)<terms)
    depth++;
  depth++;

  if (cores < 1) {
        fprintf(stderr,"Number of cores reset from %ld to 1\n",cores); 
        fflush(stderr);
	cores = 1;
  }
  if ((terms > 0) && (terms < cores)) {
        fprintf(stderr,"Number of cores reset from %ld to %ld\n",cores,terms); 
        fflush(stderr);
	cores = terms;
  }
  cores_depth = 0;
  while ((1L<<cores_depth)<cores)
    cores_depth++;
  cores_size=pow(2,cores_depth);

  percent = terms/100.0;

  fprintf(stderr,"#terms=%ld, depth=%ld, cores=%ld\n", terms, depth, cores);

  begin = mid0 = clock();

  pstack = malloc(sizeof(mpz_t)*cores);
  qstack = malloc(sizeof(mpz_t)*cores);
  gstack = malloc(sizeof(mpz_t)*cores);
  /* allocate stacks */
  for (j = 0; j < cores_size; j++) {
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
  } else {

    mid0 = clock();

    mid = terms / cores; 

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < cores; i++) {
      if (i < (cores-1))
         bs(i*mid, (i+1)*mid, 1, 0, i, 0);
      else
         bs(i*mid, terms, 1, 0, i, 0);
    }

    for (k = 1; k < cores_size; k*=2) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (i = 0; i < cores; i=i+2*k) {
        if (i+k < cores) {
          sum( i, i+k, 1);
        }
      }
    }

    for (j=0; j<cores_size; j++) {
      free(gstack[j]);
    }
    free(gstack);
  }

  mid1 = clock();
  fprintf(stderr,"\nbs      time = %6.3f\n", (double)(mid1-mid0)/CLOCKS_PER_SEC);

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

  mid2 = clock();
  //fprintf(stderr,"time = %6.3f\n", (double)(mid2-mid1)/CLOCKS_PER_SEC);

  /* initialize temp float variables for sqrt & div */
  mpf_init(t1);
  mpf_init(t2);
  //mpf_set_prec_raw(t1, mpf_get_prec(pi));

  /* final step */
  fprintf(stderr,"div     ");  fflush(stderr);
  my_div(qi, pi, qi);
  mid3 = clock();
  fprintf(stderr,"time = %6.3f\n", (double)(mid3-mid2)/CLOCKS_PER_SEC);

  fprintf(stderr,"sqrt    ");  fflush(stderr);
  my_sqrt_ui(pi, C);
  mid4 = clock();
  fprintf(stderr,"time = %6.3f\n", (double)(mid4-mid3)/CLOCKS_PER_SEC);

  fprintf(stderr,"mul     ");  fflush(stderr);
  mpf_mul(qi, qi, pi);
  end = clock();
  fprintf(stderr,"time = %6.3f\n", (double)(end-mid4)/CLOCKS_PER_SEC);

  fprintf(stderr,"total   time = %6.3f\n", (double)(end-begin)/CLOCKS_PER_SEC);
  fflush(stderr);

  fprintf(stderr,"   P size=%ld digits (%f)\n"
	 "   Q size=%ld digits (%f)\n",
	 psize, (double)psize/d, qsize, (double)qsize/d);

  /* output Pi and timing statistics */
  if (out&1)  {
    fprintf(stdout,"pi(0,%ld)=\n", terms);
    mpf_out_str(stdout, 10, d+2, qi);
    fprintf(stdout,"\n");
  }

  /* free float resources */
  mpf_clear(pi);
  mpf_clear(qi);

  mpf_clear(t1);
  mpf_clear(t2);
  exit (0);
}
