/* Pi computation using Chudnovsky's algortithm.

 * Copyright 2002, 2005 Hanhong Xue (macroxue at yahoo dot com)

 * Slightly modified 2005 by Torbjorn Granlund (tege at swox dot com) to allow
   more than 2G digits to be computed.

 * Modifed 2008 by David Carver (dcarver at tacc dot utexas dot edu) to enable
   multi-threading using the algorithm from "Fast multiprecision evaluation of series of 
   rational numbers" by Bruno Haible and Thomas Papanikolaou; and "Computation of High-Precision 
   Mathematical Constants in a Combined Cluster and Grid Environment" by 
   Daisuke Takahashi, Mitsuhisa Sato, and Taisuke Boku.

 * Modified 2010, 2024 to demonstrate a fully recursive binary splitting version using ideas 
   from "10 Trillion Digits of Pi: A Case Study of summing Hypergeometric Series to high
   precision on Multicore Systems" by Yee and Kondo at 
   https://www.ideals.illinois.edu/items/28571/bitstreams/96266/data.pdf.
   Also, simplified OpenMP and improve performance incorperating excellent ideas for nested 
   parallelism from Mario Roy implementation at 
   https://github.com/marioroy/Chudnovsky-Pi.

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

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

typedef struct {
  unsigned long max_facs;
  unsigned long num_facs;
  unsigned long *fac;
  unsigned long *pow;
} fac_t[1];

typedef struct {
  long fac;
  long pow;
  long nxt;
} sieve_t;

fac_t fpstack, fgstack;

sieve_t *sieve;
long sieve_size;

#define INIT_FACS 32

void
fac_show(fac_t f)
{
  long i;
  for (i=0; i<f[0].num_facs; i++)
    if (f[0].pow[i]==1)
      fprintf(stderr, "%ld ", f[0].fac[i]);
    else
      fprintf(stderr, "%ld^%ld ", f[0].fac[i], f[0].pow[i]);
  fprintf(stderr, "\n");
}

void
fac_reset(fac_t f)
{
  f[0].num_facs = 0;
}

void
fac_init_size(fac_t f, long s)
{
  if (s<INIT_FACS)
    s=INIT_FACS;

  f[0].fac  = malloc(s*sizeof(unsigned long)*2);
  f[0].pow  = f[0].fac + s;
  f[0].max_facs = s;

  fac_reset(f);
}

void
fac_init(fac_t f)
{
  fac_init_size(f, INIT_FACS);
}

void
fac_clear(fac_t f)
{
  free(f[0].fac);
}

void
fac_resize(fac_t f, long s)
{
  if (f[0].max_facs < s) {
    fac_clear(f);
    fac_init_size(f, s);
  }
}

// f = base^pow
void
fac_set_bp(fac_t f,unsigned long base, long pow)
{
  long i;
  assert(base<sieve_size);
  for (i=0, base/=2; base>0; i++, base = sieve[base].nxt) {
    f[0].fac[i] = sieve[base].fac;
    f[0].pow[i] = sieve[base].pow*pow;
  }
  f[0].num_facs = i;
  assert(i<=f[0].max_facs);
}

// r = f*g
void
fac_mul2(fac_t r, fac_t f, fac_t g)
{
  long i, j, k;

  for (i=j=k=0; i<f[0].num_facs && j<g[0].num_facs; k++) {
    if (f[0].fac[i] == g[0].fac[j]) {
      r[0].fac[k] = f[0].fac[i];
      r[0].pow[k] = f[0].pow[i] + g[0].pow[j];
      i++; j++;
    } else if (f[0].fac[i] < g[0].fac[j]) {
      r[0].fac[k] = f[0].fac[i];
      r[0].pow[k] = f[0].pow[i];
      i++;
    } else {
      r[0].fac[k] = g[0].fac[j];
      r[0].pow[k] = g[0].pow[j];
      j++;
    }
  }
  for (; i<f[0].num_facs; i++, k++) {
    r[0].fac[k] = f[0].fac[i];
    r[0].pow[k] = f[0].pow[i];
  }
  for (; j<g[0].num_facs; j++, k++) {
    r[0].fac[k] = g[0].fac[j];
    r[0].pow[k] = g[0].pow[j];
  }
  r[0].num_facs = k;
  assert(k<=r[0].max_facs);
}

// f *= g
void
fac_mul(fac_t f, fac_t g)
{
  fac_t tmp, fm;
  fac_init(fm);
  fac_resize(fm, f->num_facs + g->num_facs);
  fac_mul2(fm, f, g);
  tmp[0] = f[0];
  f[0]    = fm[0];
  fm[0] = tmp[0];
  fac_clear(fm);
}

// f *= base^pow
void
fac_mul_bp(fac_t f,unsigned long base,unsigned long pow)
{
  fac_t ft;
  fac_init(ft);
  fac_set_bp(ft, base, pow);
  fac_mul(f, ft);
  fac_clear(ft);
}

// remove factors of power 0
void
fac_compact(fac_t f)
{
  long i, j;
  for (i=0, j=0; i<f[0].num_facs; i++) {
    if (f[0].pow[i]>0) {
      if (j<i) {
	      f[0].fac[j] = f[0].fac[i];
	f[0].pow[j] = f[0].pow[i];
      }
      j++;
    }
  }
  f[0].num_facs = j;
}

// convert factorized form to number
void
bs_mul(mpz_t r, long a, long b, fac_t fm)
{
  long i, j;
  if (b-a<=32) {
    mpz_set_ui(r, 1);
    for (i=a; i<b; i++)
      for (j=0; j<fm->pow[i]; j++)
        mpz_mul_ui(r, r, fm->fac[i]);
  } else {
    mpz_t r2;
    mpz_init(r2);
    bs_mul(r2, a, (a+b)/2, fm);
    bs_mul(r, (a+b)/2, b, fm);
    mpz_mul(r, r, r2);
    mpz_clear(r2);
  }
}

// f /= gcd(f,g), g /= gcd(f,g)
void
fac_remove_gcd(mpz_t p, fac_t fp, mpz_t g, fac_t fg)
{
  long i, j, k, c;
  mpz_t pgcd;
  fac_t fm;
  mpz_init(pgcd);
  fac_init(fm);
  fac_resize(fm, min(fp->num_facs, fg->num_facs));
  for (i=j=k=0; i<fp->num_facs && j<fg->num_facs; ) {
    if (fp->fac[i] == fg->fac[j]) {
      c = min(fp->pow[i], fg->pow[j]);
      fp->pow[i] -= c;
      fg->pow[j] -= c;
      fm->fac[k] = fp->fac[i];
      fm->pow[k] = c;
      i++; j++; k++;
    } else if (fp->fac[i] < fg->fac[j]) {
      i++;
    } else {
      j++;
    }
  }
  fm->num_facs = k;
  assert(k <= fm->max_facs);

  if (fm->num_facs) {
    bs_mul(pgcd, 0, fm->num_facs, fm);

// Old way
//    mpz_tdiv_q(p, p, pgcd);
//    mpz_tdiv_q(g, g, pgcd);
#define SIZ(x) x->_mp_size
    mpz_divexact(p, p, pgcd);
    mpz_divexact(g, g, pgcd);
    fac_compact(fp);
    fac_compact(fg);

  }
  fac_clear(fm);
  mpz_clear(pgcd);
}

void
build_sieve(long n, sieve_t *s)
{
  long m, i, j, k, id2, jd2;

  sieve_size = n;
  m = (long)sqrt(n);
  memset(s, 0, sizeof(sieve_t)*n/2);

  s[1/2].fac = 1;
  s[1/2].pow = 1;

  for (i=3; i<=n; i+=2) {
    id2 = i >> 1;
    if (s[id2].fac == 0) {
      s[id2].fac = i;
      s[id2].pow = 1;
      if (i<=m) {
	for (j=i*i, k=id2; j<=n; j+=i+i, k++) {
          jd2 = j  >> 1;
	  if (s[jd2].fac==0) {
	    s[jd2].fac = i;
	    if (s[k].fac == i) {
	      s[jd2].pow = s[k].pow + 1;
	      s[jd2].nxt = s[k].nxt;
	    } else {
	      s[jd2].pow = 1;
	      s[jd2].nxt = k;
	    }
	  }
	}
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////

/* binary splitting */

void
bs(unsigned long a, unsigned long b, long gflag, long threads, long level, 
   mpz_t pstack1, mpz_t qstack1, mpz_t gstack1, fac_t fpstack1, fac_t fgstack1)
{
  unsigned long mid;
  long i;

  if (out&2)
  {
    fprintf(stderr, "bs: a = %ld b = %ld gflag = %ld \n", a, b, gflag);
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

    i=b;
    while ((i&1)==0) i>>=1;
    fac_set_bp(fpstack1, i, 3);    // b^3
    fac_mul_bp(fpstack1, 3*5*23*29, 3);
    fpstack1[0].pow[0]--;

    fac_set_bp(fgstack1, 2*b-1, 1);   // 2b-1
    fac_mul_bp(fgstack1, 6*b-1, 1);   // 6b-1
    fac_mul_bp(fgstack1, 6*b-5, 1);   // 6b-5

  } else {

    mpz_t pstack2, qstack2, gstack2;
    fac_t fpstack2, fgstack2;

    mpz_init(pstack2);
    mpz_init(qstack2);
    mpz_init(gstack2);
    fac_init(fpstack2);
    fac_init(fgstack2);

    /*
      p(a,b) = p(a,m) * p(m,b)
      g(a,b) = g(a,m) * g(m,b)
      q(a,b) = q(a,m) * p(m,b) + q(m,b) * g(a,m)
    */

    mid = a+(b-a)*0.5224;     /* tuning parameter */

    if (((b-a) < cutoff) || (threads < 1))
    {
      bs(a, mid, 1, threads/2, level+1, pstack1, qstack1, gstack1, fpstack1, fgstack1);
      bs(mid, b, gflag, threads/2, level+1, pstack2, qstack2, gstack2, fpstack2, fgstack2);
    } else {
          #pragma omp task default(none) shared(a, mid, threads, level, pstack1, qstack1, gstack1, fpstack1, fgstack1)
            bs(a, mid, 1, threads/2, level+1, pstack1, qstack1, gstack1, fpstack1, fgstack1);

          #pragma omp task default(none) shared(mid, b, gflag, threads, level, pstack2, qstack2, gstack2, fpstack2, fgstack2)
            bs(mid, b, gflag, threads/2, level+1, pstack2, qstack2, gstack2, fpstack2, fgstack2);
          #pragma omp taskwait 
    }

    if (level>=4) {           // tuning parameter
      fac_remove_gcd(pstack2, fpstack2, gstack1, fgstack1);
    }

    if (((b-a) < cutoff) || (threads < 3))
    {
       mpz_mul(pstack1, pstack1, pstack2);
       mpz_mul(qstack1, qstack1, pstack2);
       mpz_mul(qstack2, qstack2, gstack1);
    } else {
      #pragma omp parallel num_threads(3)
      {
        #pragma omp single nowait
        {
          #pragma omp task default(none) shared(pstack1, pstack2)
            mpz_mul(pstack1, pstack1, pstack2);

          #pragma omp task default(none) shared(qstack1, pstack2)
            mpz_mul(qstack1, qstack1, pstack2);

          #pragma omp task default(none) shared(qstack2, gstack1)
            mpz_mul(qstack2, qstack2, gstack1);

          #pragma omp taskwait 
        }
      }
    }

    mpz_clear(pstack2);
    fac_mul(fpstack1, fpstack2);
    if (gflag)
    {
      mpz_mul(gstack1, gstack1, gstack2);
      fac_mul(fgstack1, fgstack2);
    }
    mpz_clear(gstack2);
    fac_clear(fpstack2);
    fac_clear(fgstack2);
    mpz_add(qstack1, qstack1, qstack2);
    mpz_clear(qstack2);
  }
}

int
main(int argc, char *argv[])
{
  mpz_t pstack, qstack, gstack;
  mpf_t pi, qi, ci;
  long terms, depth, psize, qsize, cores;
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

  cores=omp_get_num_procs();
  omp_set_nested(1);
  omp_set_dynamic(0);

  terms = d/DIGITS_PER_ITER;
  depth = 0;
  while ((1L<<depth)<terms)
    depth++;
  depth++;

  fprintf(stderr, "#terms=%ld, depth=%ld, threads=%ld cores=%ld cutoff=%ld\n",
          terms, depth, threads, cores, cutoff);

  // omp_set_num_threads(threads);
  if (threads < 1)
  {
        fprintf(stderr, "Number of threads reset from %ld to 1\n", threads); 
        fflush(stderr);
	threads = 1;
  }
  if (terms > 0 && terms < threads && threads <= cores)
  {
        fprintf(stderr, "Number of threads reset from %ld to %ld\n", threads, terms); 
        fflush(stderr);
	threads = terms;
  }


  mid0 = begin = cpu_time();
  wmid0 = wbegin = wall_clock();

  sieve_size = max(3*5*23*29+1, terms*6);
  sieve = (sieve_t *)malloc(sizeof(sieve_t)*sieve_size/2);
  build_sieve(sieve_size, sieve);

  mid1 = cpu_time();
  wmid1 = wall_clock();

  fprintf(stderr, "sieve      cputime = %6.2f  wallclock = %6.2f   factor = %6.2f\n",
    (mid1-mid0), (wmid1-wmid0), (mid1-mid0)/(wmid1-wmid0));
  fflush(stderr);

  mpz_init(pstack);
  mpz_init(qstack);
  mpz_init(gstack);
  fac_init(fpstack);
  fac_init(fgstack);

  /* begin binary splitting process */

  if (terms<=0)
  {
    mpz_set_ui(pstack, 1);
    mpz_set_ui(qstack, 0);
    mpz_set_ui(gstack, 1);
  } else {
    #pragma omp parallel num_threads(threads)
    {
      #pragma omp single nowait
      {
        bs(0, terms, 0, threads, 0, pstack, qstack, gstack, fpstack, fgstack);
      }
    }
  }

  mpz_clear(gstack);
  free(sieve);
  fac_clear(fpstack);
  fac_clear(fgstack);

  mid1 = cpu_time();
  wmid1 = wall_clock();
  fprintf(stderr, "bs         cputime = %6.2f  wallclock = %6.2f   factor = %6.2f\n",
    (mid1-mid0), (wmid1-wmid0), (mid1-mid0)/(wmid1-wmid0));
  fflush(stderr);

  /* prepare to convert integers to floats */

  mpf_set_default_prec((long)(d*BITS_PER_DIGIT+16));

  /*
	  p*(C/D)*sqrt(C)
    pi = -----------------
	     (q+A*p)
  */

  psize = mpz_sizeinbase(pstack, 10);
  qsize = mpz_sizeinbase(qstack, 10);

  mpz_addmul_ui(qstack, pstack, A);
  mpz_mul_ui(pstack, pstack, C/D);

  mpf_init(pi);
  mpf_set_z(pi, pstack);
  mpz_clear(pstack);

  mpf_init(qi);
  mpf_set_z(qi, qstack);
  mpz_clear(qstack);

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
        #pragma omp task default(none) shared(pi, qi)
          mpf_div(qi, pi, qi);

        #pragma omp task default(none) shared(ci)
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
