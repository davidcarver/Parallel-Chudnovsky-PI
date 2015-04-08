/* Pi computation using Chudnovsky's algortithm.

 * Copyright 2002,2005 Hanhong Xue (macroxue at yahoo dot com)

 * Slightly modified 2005 by Torbjorn Granlund to allow more than 2G
   digits to be computed.

 * Modified 2015 by David Carver (dcarver at tacc dot utexas dot edu) to 
   demonstrate a parallel and fully recursive version of the gmp-chudnovsky 
   program using Cilkplus.

   To compile with gcc 5.0 or later::
   gcc -Wall -O2 -fcilkplus -o raspberry-pi raspberry-pi.c -lgmp -lm
 
   To run:
   ./raspberry-pi 1000 1

   To get help run the program with no options:
   ./raspberry-pi

 * Redistribution and use in source and binary forms,with or without
 * modification,are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES,INCLUDING,BUT NOT LIMITED TO,THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
 * EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT,INDIRECT,INCIDENTAL,
 * SPECIAL,EXEMPLARY,OR CONSEQUENTIAL DAMAGES (INCLUDING,BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,DATA,OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT,STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,EVEN IF
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
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#define A   13591409
#define B   545140134
#define C   640320
#define D   12

#define BITS_PER_DIGIT   3.32192809488736234787
#define DIGITS_PER_ITER  14.1816474627254776555
#define DOUBLE_PREC      53

long terms;
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

/* binary splitting */
void
bs(unsigned long a,unsigned long b,unsigned long level,mpz_t pstack1,mpz_t qstack1,mpz_t gstack1)
{
  unsigned long mid;
  mpz_t pstack2,qstack2,gstack2;

  if (b-a==1) {

    /*
      g(b-1,b) = (6b-5)(2b-1)(6b-1)
      p(b-1,b) = b^3 * C^3 / 24
      q(b-1,b) = (-1)^b*g(b-1,b)*(A+Bb).
    */

    mpz_set_ui(pstack1,b);
    mpz_mul_ui(pstack1,pstack1,b);
    mpz_mul_ui(pstack1,pstack1,b);
    mpz_mul_ui(pstack1,pstack1,(C/24)*(C/24));
    mpz_mul_ui(pstack1,pstack1,C*24);

    mpz_set_ui(gstack1,2*b-1);
    mpz_mul_ui(gstack1,gstack1,6*b-1);
    mpz_mul_ui(gstack1,gstack1,6*b-5);

    mpz_set_ui(qstack1,b);
    mpz_mul_ui(qstack1,qstack1,B);
    mpz_add_ui(qstack1,qstack1,A);
    mpz_mul   (qstack1,qstack1,gstack1);
    if (b%2)
      mpz_neg(qstack1,qstack1);

  } else {

    mpz_init(pstack2);
    mpz_init(qstack2);
    mpz_init(gstack2);

    if (b-a==2) {
      mpz_set_ui(pstack1,(b-1));
      mpz_mul_ui(pstack1,pstack1,(b-1));
      mpz_mul_ui(pstack1,pstack1,(b-1));
      mpz_mul_ui(pstack1,pstack1,(C/24)*(C/24));
      mpz_mul_ui(pstack1,pstack1,C*24);

      mpz_set_ui(gstack1,2*(b-1)-1);
      mpz_mul_ui(gstack1,gstack1,6*(b-1)-1);
      mpz_mul_ui(gstack1,gstack1,6*(b-1)-5);

      mpz_set_ui(qstack1,(b-1));
      mpz_mul_ui(qstack1,qstack1,B);
      mpz_add_ui(qstack1,qstack1,A);
      mpz_mul   (qstack1,qstack1,gstack1);
      if ((b-1)%2)
        mpz_neg(qstack1,qstack1);

      mpz_set_ui(pstack2,b);
      mpz_mul_ui(pstack2,pstack2,b);
      mpz_mul_ui(pstack2,pstack2,b);
      mpz_mul_ui(pstack2,pstack2,(C/24)*(C/24));
      mpz_mul_ui(pstack2,pstack2,C*24);

      mpz_set_ui(gstack2,2*b-1);
      mpz_mul_ui(gstack2,gstack2,6*b-1);
      mpz_mul_ui(gstack2,gstack2,6*b-5);

      mpz_set_ui(qstack2,b);
      mpz_mul_ui(qstack2,qstack2,B);
      mpz_add_ui(qstack2,qstack2,A);
      mpz_mul   (qstack2,qstack2,gstack2);
      if (b%2)
        mpz_neg(qstack2,qstack2);

    } else {

    /*
      p(a,b) = p(a,m) * p(m,b)
      g(a,b) = g(a,m) * g(m,b)
      q(a,b) = q(a,m) * p(m,b) + q(m,b) * g(a,m)
    */

      mid = a+(b-a)*0.5224;     /* tuning parameter */

      cilk_spawn bs(a,mid,level+1,pstack1,qstack1,gstack1);

      bs(mid,b,level+1,pstack2,qstack2,gstack2);
      cilk_sync;

    }

    mpz_mul(pstack1,pstack1,pstack2);
    mpz_mul(qstack1,qstack1,pstack2);
    mpz_addmul(qstack1,qstack2,gstack1);

    if (b < terms) {
      mpz_mul(gstack1,gstack1,gstack2);
    }

    mpz_clear(pstack2);
    mpz_clear(qstack2);
    mpz_clear(gstack2);
  }
}

void cilk_hack (mpf_t ci, mpf_t qi, mpf_t pi)
/*
 * Hack to allow __cilkrts_set_parms in main()
*/
{
  mpf_init(ci);
  cilk_spawn mpf_sqrt_ui(ci,C);
  mpf_div(qi,pi,qi);
  cilk_sync;
  mpf_clear(pi);
}

int
main(int argc,char *argv[])
{
  mpf_t  pi,qi,ci;
  mpz_t   pstack,qstack,gstack;
  long d=100,out=0,threads=1,depth,psize,qsize;
  double begin, mid0, mid3, mid4, end;
  double wbegin, wmid0, wmid3, wmid4, wend;

  prog_name = argv[0];

  if (argc==1) {
    fprintf(stderr,"\nSyntax: %s <digits> <option> <threads>\n",prog_name);
    fprintf(stderr,"      <digits> digits of pi to output\n");
    fprintf(stderr,"      <option> 0 - just run (default)\n");
    fprintf(stderr,"               1 - output digits\n");
    fprintf(stderr,"      <threads> number of threads (default 1)\n");
    exit(1);
  }
  if (argc>1)
    d = strtoul(argv[1],0,0);
  if (argc>2)
    out = atoi(argv[2]);
  if (argc>3) {
    threads = atoi(argv[3]);
    if ( threads > 0 )
      __cilkrts_end_cilk();
      __cilkrts_set_param("nworkers", argv[3]);
  }

  terms = d/DIGITS_PER_ITER;
  depth = 0;
  while ((1L<<depth)<terms)
    depth++;
  depth++;

  fprintf(stderr,"#terms=%ld, depth=%ld, threads=%ld cores=%d\n", terms, depth, __cilkrts_get_nworkers(), get_nprocs());

  begin = cpu_time();
  wbegin = wall_clock();

  mpz_init(pstack);
  mpz_init(qstack);
  mpz_init(gstack);

  /* begin binary splitting process */

  if (terms<=0) {
    mpz_set_ui(pstack,1);
    mpz_set_ui(qstack,0);
    mpz_set_ui(gstack,1);
  } else {
      bs(0,terms,1,pstack,qstack,gstack);
  }

  mid0 = cpu_time();
  wmid0 = wall_clock();
  fprintf(stderr,"bs       cputime = %6.2f  wallclock = %6.2f   factor = %6.1f\n",
    mid0-begin,wmid0-wbegin,(mid0-begin)/(wmid0-wbegin));
  fflush(stderr);

  mpz_clear(gstack);

  /* prepare to convert integers to floats */

  mpf_set_default_prec((long)(d*BITS_PER_DIGIT+16));

  /*
	  p*(C/D)*sqrt(C)
    pi = -----------------
	     (q+A*p)
  */

  psize = mpz_sizeinbase(pstack,10);
  qsize = mpz_sizeinbase(qstack,10);

  mpz_addmul_ui(qstack,pstack,A);
  mpz_mul_ui(pstack,pstack,C/D);

  mpf_init(pi);
  mpf_set_z(pi,pstack);
  mpz_clear(pstack);

  mpf_init(qi);
  mpf_set_z(qi,qstack);
  mpz_clear(qstack);

  /* final step */

  mid3 = cpu_time();
  wmid3 = wall_clock();

  cilk_hack (ci, qi, pi);

  mid4 = cpu_time();
  wmid4 = wall_clock();
  fprintf(stderr,"div/sqrt cputime = %6.2f  wallclock = %6.2f   factor = %6.1f\n",
    mid4-mid3,wmid4-wmid3,(mid4-mid3)/(wmid4-wmid3));

  mpf_mul(qi,qi,ci);
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
    mpf_out_str(stdout,10,d,qi);
    fprintf(stdout,"\n");
  }

  /* free float resources */

  mpf_clear(qi);

  exit (0);
}
