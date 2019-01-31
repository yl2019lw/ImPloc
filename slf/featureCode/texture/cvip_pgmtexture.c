/* 
 *********************************************************************
 * Copyright (C) 2006 Murphy Lab,Carnegie Mellon University
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 * 
 * For additional information visit http://murphylab.web.cmu.edu or
 * send email to murphy@cmu.edu
 */
/* pgmtxtur.c - calculate textural features on a portable graymap
**
** Author: James Darrell McCauley
**         Texas Agricultural Experiment Station
**         Department of Agricultural Engineering
**         Texas A&M University
**         College Station, Texas 77843-2117 USA
**
** Code written partially taken from pgmtofs.c in the PBMPLUS package
** by Jef Poskanzer.
**
** Algorithms for calculating features (and some explanatory comments) are
** taken from:
**
**   Haralick, R.M., K. Shanmugam, and I. Dinstein. 1973. Textural features
**   for image classification.  IEEE Transactions on Systems, Man, and
**   Cybertinetics, SMC-3(6):610-621.
**
** Copyright (C) 1991 Texas Agricultural Experiment Station, employer for
** hire of James Darrell McCauley
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
** THE TEXAS AGRICULTURAL EXPERIMENT STATION (TAES) AND THE TEXAS A&M
** UNIVERSITY SYSTEM (TAMUS) MAKE NO EXPRESS OR IMPLIED WARRANTIES
** (INCLUDING BY WAY OF EXAMPLE, MERCHANTABILITY) WITH RESPECT TO ANY
** ITEM, AND SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL
** OR CONSEQUENTAL DAMAGES ARISING OUT OF THE POSESSION OR USE OF
** ANY SUCH ITEM. LICENSEE AND/OR USER AGREES TO INDEMNIFY AND HOLD
** TAES AND TAMUS HARMLESS FROM ANY CLAIMS ARISING OUT OF THE USE OR
** POSSESSION OF SUCH ITEMS.
** 
** Modification History:
** 24 Jun 91 - J. Michael Carstensen <jmc@imsor.dth.dk> supplied fix for 
**             correlation function.
**
** Aug. 7 96 - Wenxing Li: huge memory leaks are fixed.

   23 Nov 98 - M. Boland : Compile with the following for use with Matlab
                 under Red Hat Linux 5.1 (i.e. use libc5 instead of glibc)

gcc -c -B /usr/libc5/usr/lib/gcc-lib/ -nostdinc -nostdinc++ -I/usr/libc5/usr/include -I/usr/libc5/usr/lib/gcc-lib/i386-linux/2.7.2.1/include -I/home/boland/Matlab/Mex/Include -ansi cvip_pgmtexture.c

ar r libmb_cvip.a cvip_pgmtexture.o

>> mex -f gccopts.sh -lmb_cvip -L/home/boland/Matlab/Mex  mb_texture.c


   29 Nov 98 - M. Boland : Modified calculations to produce the same values
                 as the kharalick routine in Khoros.  Some feature calculations
                 below were wrong, others made different assumptions (the
                 Haralick paper is not always explicit).

*/

#include <math.h>
#include "ppgm.h"
#include "CVIPtexture.h"

#define RADIX 2.0
#define EPSILON 0.000000001
#define BL  "Angle                 "
#define F1  "Angular Second Moment "
#define F2  "Contrast              "
#define F3  "Correlation           "
#define F4  "Variance              "
#define F5  "Inverse Diff Moment   "
#define F6  "Sum Average           "
#define F7  "Sum Variance          "
#define F8  "Sum Entropy           "
#define F9  "Entropy               "
#define F10 "Difference Variance   "
#define F11 "Difference Entropy    "
#define F12 "Meas of Correlation-1 "
#define F13 "Meas of Correlation-2 "
#define F14 "Max Correlation Coeff "

#define SIGN(x,y) ((y)<0 ? -fabs(x) : fabs(x))
#define DOT fprintf(stderr,".")
#define SWAP(a,b) {y=(a);(a)=(b);(b)=y;}


 

void results (),  mkbalanced (), reduction (), simplesrt ();
int hessenberg ();
float f1_asm (), f2_contrast (), f3_corr (), f4_var (), f5_idm (),
 f6_savg (), f7_svar (), f8_sentropy (), f9_entropy (), f10_dvar (),
 f11_dentropy (), f12_icorr (), f13_icorr (), f14_maxcorr (), *pgm_vector (),
 **pgm_matrix ();



TEXTURE * Extract_Texture_Features(int distance, register gray **grays, int rows, int cols, TEXTURE_FEATURE_MAP *feature_usage)  
{
  FILE *ifp;
  register gray  *gP;
  int tone[PGM_MAXMAXVAL+1], R0, R45, R90, R135, angle, d = 1, x, y;
  int argn, bps, padright, row, col, i;
  int itone, jtone, tones,g_val;
  float **P_matrix0, **P_matrix45, **P_matrix90, **P_matrix135;
  float ASM[4], contrast[4], corr[4], var[4], idm[4], savg[4];
  float sentropy[4], svar[4], entropy[4], dvar[4], dentropy[4];
  float icorr[4], maxcorr[4];
  float *Tp;
  gray nmaxval, maxval;
  char *usage = "[-d <d>] [pgmfile]";
  TEXTURE *Texture;

    Texture = (TEXTURE *) calloc(1,sizeof(TEXTURE));
    if(!Texture) {
        printf("\nERROR in TEXTURE structure allocate\n");
        exit(1);
    }

  d = distance; 

   /* Determine the number of different gray scales (not maxval) */
  for (row = PGM_MAXMAXVAL; row >= 0; --row)
    tone[row] = -1;
  for (row = rows - 1; row >= 0; --row)
    for (col = 0; col < cols; ++col)
      {
   /*   if (grays[row][col])   If gray value equal 0 don't include */		
        tone[grays[row][col]] = grays[row][col];
      }	
  
 for (row = PGM_MAXMAXVAL, tones = 0; row >= 0; --row)
    if (tone[row] != -1)
      tones++;
 /* fprintf (stderr, "(Image has %d graylevels.)\n", tones); */

  /* Collapse array, taking out all zero values */
  for (row = 0, itone = 0; row <= PGM_MAXMAXVAL; row++)
    if (tone[row] != -1)
      tone[itone++] = tone[row];
  /* Now array contains only the gray levels present (in ascending order) */

  /* Allocate memory for gray-tone spatial dependence matrix */
  P_matrix0 = pgm_matrix (0, tones, 0, tones);
  P_matrix45 = pgm_matrix (0, tones, 0, tones);
  P_matrix90 = pgm_matrix (0, tones, 0, tones);
  P_matrix135 = pgm_matrix (0, tones, 0, tones);
  for (row = 0; row < tones; ++row)
    for (col = 0; col < tones; ++col)
    {
      P_matrix0[row][col] = P_matrix45[row][col] = 0;
      P_matrix90[row][col] = P_matrix135[row][col] = 0;
    }

   R0 = 0;
   R45 = 0;
   R90 = 0;
   R135 = 0;	
  /* Find gray-tone spatial dependence matrix */
 /* fprintf (stderr, "(Computing spatial dependence matrix..."); */
 
  for (row = 0; row < rows; ++row)
    for (col = 0; col < cols; ++col)
      if (grays[row][col])  /* if value anything other than zero */
      for (x = 0, angle = 0; angle <= 135; angle += 45)
      {
	while (tone[x] != grays[row][col])
	  x++;
	/* M. Boland if (angle == 0 && col + d < cols)  */
	/* M. Boland - include neighbor only if != 0 */
	if (angle == 0 && col + d < cols && grays[row][col + d]) 
	{
	  y = 0;
	  while (tone[y] != grays[row][col + d])
	    y++;
  	  P_matrix0[x][y]++;
 	  P_matrix0[y][x]++;
  	  /* R0++;  M. Boland 25 Nov 98 */
	  R0+=2 ;
	}
	/* M. Boland if (angle == 90 && row + d < rows) */
	/* M. Boland - include neighbor only if != 0 */
	else if (angle == 90 && row + d < rows && grays[row + d][col])
	{
	  y = 0;
	  while (tone[y] != grays[row + d][col])
	    y++;
	    P_matrix90[x][y]++;
	    P_matrix90[y][x]++;
   	    /* R90++;  M. Boland 25 Nov 98 */
	    R90+=2 ;
	}
	/* M. Boland if (angle == 45 && row + d < rows && col - d >= 0) */
	/* M. Boland - include neighbor only if != 0 */
	else if (angle == 45 && row + d < rows && col - d >= 0 
		 && grays[row + d][col - d])
	{
	  y = 0;
	  while (tone[y] != grays[row + d][col - d])
	    y++;
  	  P_matrix45[x][y]++;
	  P_matrix45[y][x]++;
	  /* R45++;  M. Boland 25 Nov 98 */
	  R45+=2 ;
	}
	/* M. Boland if (angle == 135 && row + d < rows && col + d < cols) */
	else if (angle == 135 && row + d < rows && col + d < cols 
		 && grays[row + d][col + d])
	{
	  y = 0;
	  while (tone[y] != grays[row + d][col + d])
	    y++;
	  P_matrix135[x][y]++;
	  P_matrix135[y][x]++;
	  /* R135++;  M. Boland 25 Nov 98 */
	  R135+=2 ;
	}

     }	
  /* Gray-tone spatial dependence matrices are complete */

  /* Find normalizing constants */
/* R0 = 2 * rows * (cols - 1);
  R45 = 2 * (rows - 1) * (cols - 1);
  R90 = 2 * (rows - 1) * cols;
  R135 = R45;
*/

  /* Normalize gray-tone spatial dependence matrix */
  for (itone = 0; itone < tones; ++itone)
    for (jtone = 0; jtone < tones; ++jtone)
    {
      P_matrix0[itone][jtone] /= R0;
      P_matrix45[itone][jtone] /= R45;
      P_matrix90[itone][jtone] /= R90;
      P_matrix135[itone][jtone] /= R135;
    }

/*  fprintf (stderr, " done.)\n"); */
/*  fprintf (stderr, "(Computing textural features"); */
/*  fprintf (stdout, "\n"); */
/*  DOT; */
/*  fprintf (stdout,
	   "%s         0         45         90        135        Avg       Range\n",
	   BL);
*/
  if (feature_usage->ASM)
	{
  	ASM[0] = f1_asm (P_matrix0, tones);
  	ASM[1] = f1_asm (P_matrix45, tones);
  	ASM[2] = f1_asm (P_matrix90, tones);
  	ASM[3] = f1_asm (P_matrix135, tones);
	}
  Tp = &Texture->ASM[0];
  results (Tp, F1, ASM);

  if (feature_usage->contrast)
	{
  	contrast[0] = f2_contrast (P_matrix0, tones);
  	contrast[1] = f2_contrast (P_matrix45, tones);
  	contrast[2] = f2_contrast (P_matrix90, tones);
  	contrast[3] = f2_contrast (P_matrix135, tones);
	}
  Tp = &Texture->contrast[0];
  results (Tp, F2, contrast);

  if (feature_usage->correlation)
	{
 	corr[0] = f3_corr (P_matrix0, tones);
 	corr[1] = f3_corr (P_matrix45, tones);
  	corr[2] = f3_corr (P_matrix90, tones);
  	corr[3] = f3_corr (P_matrix135, tones);
	}
  Tp = &Texture->correlation[0];
  results (Tp, F3, corr);

  if (feature_usage->variance)
	{
  	var[0] = f4_var (P_matrix0, tones);
  	var[1] = f4_var (P_matrix45, tones);
  	var[2] = f4_var (P_matrix90, tones);
  	var[3] = f4_var (P_matrix135, tones);
	}
  Tp = &Texture->variance[0];
  results (Tp, F4, var); 

  if (feature_usage->IDM)
	{
  	idm[0] = f5_idm (P_matrix0, tones);
  	idm[1] = f5_idm (P_matrix45, tones);
  	idm[2] = f5_idm (P_matrix90, tones);
  	idm[3] = f5_idm (P_matrix135, tones);
	}
  Tp = &Texture->IDM[0];
  results (Tp, F5, idm); 

  if (feature_usage->sum_avg)
	{
  	savg[0] = f6_savg (P_matrix0, tones);
  	savg[1] = f6_savg (P_matrix45, tones);
  	savg[2] = f6_savg (P_matrix90, tones);
  	savg[3] = f6_savg (P_matrix135, tones);
	}
  Tp = &Texture->sum_avg[0];
  results (Tp, F6, savg); 

  if (feature_usage->sum_var)
	{
	sentropy[0] = f8_sentropy (P_matrix0, tones);
  	sentropy[1] = f8_sentropy (P_matrix45, tones);
  	sentropy[2] = f8_sentropy (P_matrix90, tones);
  	sentropy[3] = f8_sentropy (P_matrix135, tones);
	}
  if (feature_usage->sum_entropy)
	{  
	svar[0] = f7_svar (P_matrix0, tones, sentropy[0]);
  	svar[1] = f7_svar (P_matrix45, tones, sentropy[1]);
  	svar[2] = f7_svar (P_matrix90, tones, sentropy[2]);
  	svar[3] = f7_svar (P_matrix135, tones, sentropy[3]);
	}
  Tp = &Texture->sum_var[0];
  results (Tp, F7, svar); 
  Tp = &Texture->sum_entropy[0];
  results (Tp, F8, sentropy); 

  if (feature_usage->entropy)
	{
  	entropy[0] = f9_entropy (P_matrix0, tones);
  	entropy[1] = f9_entropy (P_matrix45, tones);
  	entropy[2] = f9_entropy (P_matrix90, tones);
  	entropy[3] = f9_entropy (P_matrix135, tones);
	}
  Tp = &Texture->entropy[0];
  results (Tp, F9, entropy); 

  if (feature_usage->diff_var)
	{  
	dvar[0] = f10_dvar (P_matrix0, tones);
 	dvar[1] = f10_dvar (P_matrix45, tones);
  	dvar[2] = f10_dvar (P_matrix90, tones);
  	dvar[3] = f10_dvar (P_matrix135, tones);
	}
  Tp = &Texture->diff_var[0];
  results (Tp, F10, dvar);

  if (feature_usage->diff_entropy)
	{  
	dentropy[0] = f11_dentropy (P_matrix0, tones);
  	dentropy[1] = f11_dentropy (P_matrix45, tones);
  	dentropy[2] = f11_dentropy (P_matrix90, tones);
  	dentropy[3] = f11_dentropy (P_matrix135, tones);
	}
  Tp = &Texture->diff_entropy[0];
  results (Tp, F11, dentropy);

   if (feature_usage->meas_corr1)
	{ 
	icorr[0] = f12_icorr (P_matrix0, tones);
  	icorr[1] = f12_icorr (P_matrix45, tones);
  	icorr[2] = f12_icorr (P_matrix90, tones);
  	icorr[3] = f12_icorr (P_matrix135, tones);
	}
  Tp = &Texture->meas_corr1[0];
  results (Tp, F12, icorr);

  if (feature_usage->meas_corr2)
	{  
	icorr[0] = f13_icorr (P_matrix0, tones);
  	icorr[1] = f13_icorr (P_matrix45, tones);
  	icorr[2] = f13_icorr (P_matrix90, tones);
  	icorr[3] = f13_icorr (P_matrix135, tones);
	}
  Tp = &Texture->meas_corr2[0];
  results (Tp, F13, icorr);

  if (feature_usage->max_corr_coef)
	{  
	maxcorr[0] = f14_maxcorr (P_matrix0, tones);
	maxcorr[1] = f14_maxcorr (P_matrix45, tones);
	maxcorr[2] = f14_maxcorr (P_matrix90, tones);
        maxcorr[3] = f14_maxcorr (P_matrix135, tones);
	}
  /* M. Boland - 24 Nov 98 */
  else {
    maxcorr[0] = 0 ;
    maxcorr[1] = 0 ;
    maxcorr[2] = 0 ;
    maxcorr[3] = 0 ;
  }

  Tp = &Texture->max_corr_coef[0];
  results (Tp, F14, maxcorr);

  for (i=0; i<=tones; i++) free(P_matrix0[i]);
  for (i=0; i<=tones; i++) free(P_matrix45[i]);
  for (i=0; i<=tones; i++) free(P_matrix90[i]);
  for (i=0; i<=tones; i++) free(P_matrix135[i]);

  free(P_matrix0);
  free(P_matrix45);
  free(P_matrix90);
  free(P_matrix135);

/*  fprintf (stderr, " done.)\n"); */
  return (Texture);
 /* exit (0);*/
}

float f1_asm (P, Ng)
  float **P;
  int Ng;

/* Angular Second Moment */
{
  int i, j;
  float sum = 0;

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      sum += P[i][j] * P[i][j];

  return sum;

  /*
   * The angular second-moment feature (ASM) f1 is a measure of homogeneity
   * of the image. In a homogeneous image, there are very few dominant
   * gray-tone transitions. Hence the P matrix for such an image will have
   * fewer entries of large magnitude.
   */
}


float f2_contrast (P, Ng)
  float **P;
  int Ng;

/* Contrast */
{
  int i, j, n;
  float sum = 0, bigsum = 0;

  for (n = 0; n < Ng; ++n)
  {
    for (i = 0; i < Ng; ++i)
      for (j = 0; j < Ng; ++j)
	if ((i - j) == n || (j - i) == n)
	  sum += P[i][j];
    bigsum += n * n * sum;

    sum = 0;
  }
  return bigsum;

  /*
   * The contrast feature is a difference moment of the P matrix and is a
   * measure of the contrast or the amount of local variations present in an
   * image.
   */
}

float f3_corr (P, Ng)
  float **P;
  int Ng;

/* Correlation */
{
  int i, j;
  float sum_sqrx = 0, sum_sqry = 0, tmp, *px;
  float meanx =0 , meany = 0 , stddevx, stddevy;

  px = pgm_vector (0, Ng);
  for (i = 0; i < Ng; ++i)
    px[i] = 0;

  /*
   * px[i] is the (i-1)th entry in the marginal probability matrix obtained
   * by summing the rows of p[i][j]
   */
  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      px[i] += P[i][j];


  /* Now calculate the means and standard deviations of px and py */
  /*- fix supplied by J. Michael Christensen, 21 Jun 1991 */
  /*- further modified by James Darrell McCauley, 16 Aug 1991 
   *     after realizing that meanx=meany and stddevx=stddevy
   */
  for (i = 0; i < Ng; ++i)
  {
    meanx += px[i]*i;
    sum_sqrx += px[i]*i*i;
  }
  /* M. Boland meanx = meanx/(sqrt(Ng)); */
  meany = meanx;
  sum_sqry = sum_sqrx;
  stddevx = sqrt (sum_sqrx - (meanx * meanx));
  stddevy = stddevx;

  /* Finally, the correlation ... */
  for (tmp = 0, i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      tmp += i*j*P[i][j];

 free(px); 
 return (tmp - meanx * meany) / (stddevx * stddevy);
  /*
   * This correlation feature is a measure of gray-tone linear-dependencies
   * in the image.
   */
}


float f4_var (P, Ng)
  float **P;
  int Ng;

/* Sum of Squares: Variance */
{
  int i, j;
  float mean = 0, var = 0;

  /*- Corrected by James Darrell McCauley, 16 Aug 1991
   *  calculates the mean intensity level instead of the mean of
   *  cooccurrence matrix elements 
   */
  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      mean += i * P[i][j];

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      /*  M. Boland - var += (i + 1 - mean) * (i + 1 - mean) * P[i][j]; */
      var += (i - mean) * (i - mean) * P[i][j];

  return var;
}

float f5_idm (P, Ng)
  float **P;
  int Ng;

/* Inverse Difference Moment */
{
  int i, j;
  float idm = 0;

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      idm += P[i][j] / (1 + (i - j) * (i - j));

  return idm;
}

float Pxpy[2 * PGM_MAXMAXVAL];

float f6_savg (P, Ng)
  float **P;
  int Ng;

/* Sum Average */
{
  int i, j;
  extern float Pxpy[2 * PGM_MAXMAXVAL];
  float savg = 0;

  for (i = 0; i <= 2 * Ng; ++i)
    Pxpy[i] = 0;

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      /* M. Boland Pxpy[i + j + 2] += P[i][j]; */
      /* Indexing from 2 instead of 0 is inconsistent with rest of code*/
      Pxpy[i + j] += P[i][j];
  /* M. Boland for (i = 2; i <= 2 * Ng; ++i) */
  /* Indexing from 2 instead of 0 is inconsistent with rest of code*/
  for (i = 0; i <= (2 * Ng - 2); ++i)
    savg += i * Pxpy[i];

  return savg;
}


float f7_svar (P, Ng, S)
  float **P, S;
  int Ng;

/* Sum Variance */
{
  int i, j;
  extern float Pxpy[2 * PGM_MAXMAXVAL];
  float var = 0;

  for (i = 0; i <= 2 * Ng; ++i)
    Pxpy[i] = 0;

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      /* M. Boland Pxpy[i + j + 2] += P[i][j]; */
      /* Indexing from 2 instead of 0 is inconsistent with rest of code*/
      Pxpy[i + j] += P[i][j];

  /*  M. Boland for (i = 2; i <= 2 * Ng; ++i) */
  /* Indexing from 2 instead of 0 is inconsistent with rest of code*/
  for (i = 0; i <= (2 * Ng - 2); ++i)
    var += (i - S) * (i - S) * Pxpy[i];

  return var;
}

float f8_sentropy (P, Ng)
  float **P;
  int Ng;

/* Sum Entropy */
{
  int i, j;
  extern float Pxpy[2 * PGM_MAXMAXVAL];
  float sentropy = 0;

  for (i = 0; i <= 2 * Ng; ++i)
    Pxpy[i] = 0;

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      Pxpy[i + j + 2] += P[i][j];

  for (i = 2; i <= 2 * Ng; ++i)
    /*  M. Boland  sentropy -= Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
    sentropy -= Pxpy[i] * log10 (Pxpy[i] + EPSILON)/log10(2.0) ;

  return sentropy;
}


float f9_entropy (P, Ng)
  float **P;
  int Ng;

/* Entropy */
{
  int i, j;
  float entropy = 0;

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      /*      entropy += P[i][j] * log10 (P[i][j] + EPSILON); */
      entropy += P[i][j] * log10 (P[i][j] + EPSILON)/log10(2.0) ;

  return -entropy; 
}


float f10_dvar (P, Ng)
  float **P;
  int Ng;

/* Difference Variance */
{
  int i, j, tmp;
  extern float Pxpy[2 * PGM_MAXMAXVAL];
  float sum = 0, sum_sqr = 0, var = 0;

  for (i = 0; i <= 2 * Ng; ++i)
    Pxpy[i] = 0;

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      Pxpy[abs (i - j)] += P[i][j];

  /* Now calculate the variance of Pxpy (Px-y) */
  for (i = 0; i < Ng; ++i)
  {
    sum += i * Pxpy[i] ;
    sum_sqr += i * i * Pxpy[i] ;
    /* M. Boland sum += Pxpy[i];
    sum_sqr += Pxpy[i] * Pxpy[i];*/
  }
  /*tmp = Ng * Ng ;  M. Boland - wrong anyway, should be Ng */
  /*var = ((tmp * sum_sqr) - (sum * sum)) / (tmp * tmp); */
  
  var = sum_sqr - sum*sum ;

  return var;
}

float f11_dentropy (P, Ng)
  float **P;
  int Ng;

/* Difference Entropy */
{
  int i, j, tmp;
  extern float Pxpy[2 * PGM_MAXMAXVAL];
  float sum = 0, sum_sqr = 0, var = 0;

  for (i = 0; i <= 2 * Ng; ++i)
    Pxpy[i] = 0;

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
      Pxpy[abs (i - j)] += P[i][j];

  for (i = 0; i < Ng; ++i)
    /*    sum += Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
    sum += Pxpy[i] * log10 (Pxpy[i] + EPSILON)/log10(2.0) ;

  return -sum;
}

float f12_icorr (P, Ng)
  float **P;
  int Ng;

/* Information Measures of Correlation */
/* All /log10(2.0) added by M. Boland */
{
  int i, j, tmp;
  float *px, *py;
  float hx = 0, hy = 0, hxy = 0, hxy1 = 0, hxy2 = 0;

  px = pgm_vector (0, Ng);
  py = pgm_vector (0, Ng);

  /*
   * px[i] is the (i-1)th entry in the marginal probability matrix obtained
   * by summing the rows of p[i][j]
   */
  for (i = 0; i < Ng; ++i)
  {
    for (j = 0; j < Ng; ++j)
    {
      px[i] += P[i][j];
      py[j] += P[i][j];
    }
  }

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
    {
      hxy1 -= P[i][j] * log10 (px[i] * py[j] + EPSILON)/log10(2.0);
      hxy2 -= px[i] * py[j] * log10 (px[i] * py[j] + EPSILON)/log10(2.0);
      hxy -= P[i][j] * log10 (P[i][j] + EPSILON)/log10(2.0);
    }

  /* Calculate entropies of px and py - is this right? */
  for (i = 0; i < Ng; ++i)
  {
    hx -= px[i] * log10 (px[i] + EPSILON)/log10(2.0);
    hy -= py[i] * log10 (py[i] + EPSILON)/log10(2.0);
  }
/*  fprintf(stderr,"hxy1=%f\thxy=%f\thx=%f\thy=%f\n",hxy1,hxy,hx,hy); */
  free(px);
  free(py);
  return ((hxy - hxy1) / (hx > hy ? hx : hy));
}

float f13_icorr (P, Ng)
  float **P;
  int Ng;

/* Information Measures of Correlation */
/* All /log10(2.0) added by M. Boland */

{
  int i, j;
  float *px, *py;
  float hx = 0, hy = 0, hxy = 0, hxy1 = 0, hxy2 = 0;

  px = pgm_vector (0, Ng);
  py = pgm_vector (0, Ng);

  /*
   * px[i] is the (i-1)th entry in the marginal probability matrix obtained
   * by summing the rows of p[i][j]
   */
  for (i = 0; i < Ng; ++i)
  {
    for (j = 0; j < Ng; ++j)
    {
      px[i] += P[i][j];
      py[j] += P[i][j];
    }
  }

  for (i = 0; i < Ng; ++i)
    for (j = 0; j < Ng; ++j)
    {
      hxy1 -= P[i][j] * log10 (px[i] * py[j] + EPSILON)/log10(2.0);
      hxy2 -= px[i] * py[j] * log10 (px[i] * py[j] + EPSILON)/log10(2.0);
      hxy -= P[i][j] * log10 (P[i][j] + EPSILON)/log10(2.0);
    }

  /* Calculate entropies of px and py */
  for (i = 0; i < Ng; ++i)
  {
    hx -= px[i] * log10 (px[i] + EPSILON)/log10(2.0);
    hy -= py[i] * log10 (py[i] + EPSILON)/log10(2.0);
  }
/*  fprintf(stderr,"hx=%f\thxy2=%f\n",hx,hxy2); */
  free(px);
  free(py);
  return (sqrt (abs (1 - exp (-2.0 * (hxy2 - hxy)))));
}

float f14_maxcorr (P, Ng)
  float **P;
  int Ng;

/* Returns the Maximal Correlation Coefficient */
{
  int i, j, k;
  float *px, *py, **Q;
  float *x, *iy, tmp;
  float f;

  px = pgm_vector (0, Ng);
  py = pgm_vector (0, Ng);
  Q = pgm_matrix (1, Ng + 1, 1, Ng + 1);
  x = pgm_vector (1, Ng);
  iy = pgm_vector (1, Ng);

  /*
   * px[i] is the (i-1)th entry in the marginal probability matrix obtained
   * by summing the rows of p[i][j]
   */
  for (i = 0; i < Ng; ++i)
  {
    for (j = 0; j < Ng; ++j)
    {
      px[i] += P[i][j];
      py[j] += P[i][j];
    }
  }

  /* Find the Q matrix */
  for (i = 0; i < Ng; ++i)
  {
    for (j = 0; j < Ng; ++j)
    {
      Q[i + 1][j + 1] = 0;
      for (k = 0; k < Ng; ++k)
	Q[i + 1][j + 1] += P[i][k] * P[j][k] / px[i] / py[k];
    }
  }

  /* Balance the matrix */
  mkbalanced (Q, Ng);
  /* Reduction to Hessenberg Form */
  reduction (Q, Ng);
  /* Finding eigenvalue for nonsymetric matrix using QR algorithm */
  if (!hessenberg (Q, Ng, x, iy))
	{ for (i=1; i<=Ng+1; i++) free(Q[i]+1);
	  free(Q+1);
	  free((char *)px);
	  free((char *)py);
	  free((x+1));
	  free((iy+1));
	  return 0.0;
	  /* fixed for Linux porting,
	   * I don't know what should be returned
	   */
	}
  /* simplesrt(Ng,x); */
  /* Returns the sqrt of the second largest eigenvalue of Q */
  for (i = 2, tmp = x[1]; i <= Ng; ++i)
    tmp = (tmp > x[i]) ? tmp : x[i];

  f = sqrt(x[Ng - 1]);

 for (i=1; i<=Ng+1; i++) free(Q[i]+1);
 free(Q+1);
 free((char *)px); 
 free((char *)py); 
 free((x+1)); 
 free((iy+1)); 

 return f;
}

float *pgm_vector (nl, nh)
  int nl, nh;
{
  float *v;
  int    i;

  v = (float *) malloc ((unsigned) (nh - nl + 1) * sizeof (float));
  if (!v)
    fprintf (stderr, "memory allocation failure (pgm_vector) "), exit (1);

  for (i=0; i<=(nh-nl); i++) v[i]=0;
  return v - nl;
}


float **pgm_matrix (nrl, nrh, ncl, nch)
  int nrl, nrh, ncl, nch;

/* Allocates a float matrix with range [nrl..nrh][ncl..nch] */
{
  int i;
  float **m;

  /* allocate pointers to rows */
  m = (float **) malloc ((unsigned) (nrh - nrl + 1) * sizeof (float *));
  if (!m)
    fprintf (stderr, "memory allocation failure (pgm_matrix 1) "), exit (1);
  m -= ncl;

  /* allocate rows and set pointers to them */
  for (i = nrl; i <= nrh; i++)
  {
    m[i] = (float *) malloc ((unsigned) (nch - ncl + 1) * sizeof (float));
    if (!m[i])
      fprintf (stderr, "memory allocation failure (pgm_matrix 2) "), exit (2);
    m[i] -= ncl;
  }
  /* return pointer to array of pointers to rows */
  return m;
}

void results (Tp, c, a)
  float *Tp;
  char *c;
  float *a;
{
  int i;
  float max, min;
  max = a[0];
  min = a[0];
/*  DOT;
  fprintf (stdout, "%s", c);
*/  for (i = 0; i < 4; ++i, *Tp++)
    {	
    if (a[i] <= min)
	min = a[i];
    if (a[i] > max)
	max = a[i];
  /*  fprintf (stdout, "% 1.3e ", a[i]); */
    *Tp = a[i];
    }	
/*  fprintf (stdout, "% 1.3e  % 1.3e\n", (a[0] + a[1] + a[2] + a[3]) / 4,max-min); */
  *Tp = (a[0] + a[1] + a[2] + a[3]) / 4;
  *Tp++;
  *Tp = max - min;
 
  	
}

void simplesrt (n, arr)
  int n;
  float arr[];
{
  int i, j;
  float a;

  for (j = 2; j <= n; j++)
  {
    a = arr[j];
    i = j - 1;
    while (i > 0 && arr[i] > a)
    {
      arr[i + 1] = arr[i];
      i--;
    }
    arr[i + 1] = a;
  }
}

void mkbalanced (a, n)
  float **a;
  int n;
{
  int last, j, i;
  float s, r, g, f, c, sqrdx;

  sqrdx = RADIX * RADIX;
  last = 0;
  while (last == 0)
  {
    last = 1;
    for (i = 1; i <= n; i++)
    {
      r = c = 0.0;
      for (j = 1; j <= n; j++)
	if (j != i)
	{
	  c += fabs (a[j][i]);
	  r += fabs (a[i][j]);
	}
      if (c && r)
      {
	g = r / RADIX;
	f = 1.0;
	s = c + r;
	while (c < g)
	{
	  f *= RADIX;
	  c *= sqrdx;
	}
	g = r * RADIX;
	while (c > g)
	{
	  f /= RADIX;
	  c /= sqrdx;
	}
	if ((c + r) / f < 0.95 * s)
	{
	  last = 0;
	  g = 1.0 / f;
	  for (j = 1; j <= n; j++)
	    a[i][j] *= g;
	  for (j = 1; j <= n; j++)
	    a[j][i] *= f;
	}
      }
    }
  }
}


void reduction (a, n)
  float **a;
  int n;
{
  int m, j, i;
  float y, x;

  for (m = 2; m < n; m++)
  {
    x = 0.0;
    i = m;
    for (j = m; j <= n; j++)
    {
      if (fabs (a[j][m - 1]) > fabs (x))
      {
	x = a[j][m - 1];
	i = j;
      }
    }
    if (i != m)
    {
      for (j = m - 1; j <= n; j++)
	SWAP (a[i][j], a[m][j])  
	for (j = 1; j <= n; j++)
	  SWAP (a[j][i], a[j][m]) 
	  a[j][i] = a[j][i];
    }
    if (x)
    {
      for (i = m + 1; i <= n; i++)
      {
	if (y = a[i][m - 1])
	{
	  y /= x;
	  a[i][m - 1] = y;
	  for (j = m; j <= n; j++)
	    a[i][j] -= y * a[m][j];
	  for (j = 1; j <= n; j++)
	    a[j][m] += y * a[j][i];
	}
      }
    }
  }
}

int hessenberg (a, n, wr, wi)
  float **a, wr[], wi[];
  int n;

{
  int nn, m, l, k, j, its, i, mmin;
  float z, y, x, w, v, u, t, s, r, q, p, anorm;

  anorm = fabs (a[1][1]);
  for (i = 2; i <= n; i++)
    for (j = (i - 1); j <= n; j++)
      anorm += fabs (a[i][j]);
  nn = n;
  t = 0.0;
  while (nn >= 1)
  {
    its = 0;
    do
    {
      for (l = nn; l >= 2; l--)
      {
	s = fabs (a[l - 1][l - 1]) + fabs (a[l][l]);
	if (s == 0.0)
	  s = anorm;
	if ((float) (fabs (a[l][l - 1]) + s) == s)
	  break;
      }
      x = a[nn][nn];
      if (l == nn)
      {
	wr[nn] = x + t;
	wi[nn--] = 0.0;
      }
      else
      {
	y = a[nn - 1][nn - 1];
	w = a[nn][nn - 1] * a[nn - 1][nn];
	if (l == (nn - 1))
	{
	  p = 0.5 * (y - x);
	  q = p * p + w;
	  z = sqrt (fabs (q));
	  x += t;
	  if (q >= 0.0)
	  {
	    z = p + SIGN (z, p); 
	    wr[nn - 1] = wr[nn] = x + z;
	    if (z)
	      wr[nn] = x - w / z;
	    wi[nn - 1] = wi[nn] = 0.0;
	  }
	  else
	  {
	    wr[nn - 1] = wr[nn] = x + p;
	    wi[nn - 1] = -(wi[nn] = z);
	  }
	  nn -= 2;
	}
	else
	{
	  if (its == 30)
	    {
/*	    fprintf (stderr, 
                    "Too many iterations to required to find %s\ngiving up\n", 
                     F14);  */
	     return 0; /*exit (1);*/
	     }			
	  if (its == 10 || its == 20)
	  {
	    t += x;
	    for (i = 1; i <= nn; i++)
	      a[i][i] -= x;
	    s = fabs (a[nn][nn - 1]) + fabs (a[nn - 1][nn - 2]);
	    y = x = 0.75 * s;
	    w = -0.4375 * s * s;
	  }
	  ++its;
	  for (m = (nn - 2); m >= l; m--)
	  {
	    z = a[m][m];
	    r = x - z;
	    s = y - z;
	    p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
	    q = a[m + 1][m + 1] - z - r - s;
	    r = a[m + 2][m + 1];
	    s = fabs (p) + fabs (q) + fabs (r);
	    p /= s;
	    q /= s;
	    r /= s;
	    if (m == l)
	      break;
	    u = fabs (a[m][m - 1]) * (fabs (q) + fabs (r));
	    v = fabs (p) * (fabs (a[m - 1][m - 1]) + fabs (z) + fabs (a[m + 1][m + 1]));
	    if ((float) (u + v) == v)
	      break;
	  }
	  for (i = m + 2; i <= nn; i++)
	  {
	    a[i][i - 2] = 0.0;
	    if (i != (m + 2))
	      a[i][i - 3] = 0.0;
	  }
	  for (k = m; k <= nn - 1; k++)
	  {
	    if (k != m)
	    {
	      p = a[k][k - 1];
	      q = a[k + 1][k - 1];
	      r = 0.0;
	      if (k != (nn - 1))
		r = a[k + 2][k - 1];
	      if (x = fabs (p) + fabs (q) + fabs (r))
	      {
		p /= x;
		q /= x;
		r /= x;
	      }
	    }
	    if (s = SIGN (sqrt (p * p + q * q + r * r), p)) 
	    {
	      if (k == m)
	      {
		if (l != m)
		  a[k][k - 1] = -a[k][k - 1];
	      }
	      else
		a[k][k - 1] = -s * x;
	      p += s;
	      x = p / s;
	      y = q / s;
	      z = r / s;
	      q /= p;
	      r /= p;
	      for (j = k; j <= nn; j++)
	      {
		p = a[k][j] + q * a[k + 1][j];
		if (k != (nn - 1))
		{
		  p += r * a[k + 2][j];
		  a[k + 2][j] -= p * z;
		}
		a[k + 1][j] -= p * y;
		a[k][j] -= p * x;
	      }
	      mmin = nn < k + 3 ? nn : k + 3;
	      for (i = l; i <= mmin; i++)
	      {
		p = x * a[i][k] + y * a[i][k + 1];
		if (k != (nn - 1))
		{
		  p += z * a[i][k + 2];
		  a[i][k + 2] -= p * r;
		}
		a[i][k + 1] -= p * q;
		a[i][k] -= p;
	      }
	    }
	  }
	}
      }
    } while (l < nn - 1);
  }
return 1;
}
