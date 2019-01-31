/***************************************************************************
* ======================================================================
* Computer Vision/Image Processing Tool Project - Dr. Scott Umbaugh SIUE
* ======================================================================
*
*             File Name: CVIPdef.h
*           Description: contains useful macros and constants used throughout
*			 CVIPtools.
*         Related Files: any CVIPtools source file that uses macros
*			 this header
*   Initial Coding Date: 1/3/93
*           Portability: Standard (ANSI) C
*             Credit(s): Gregory Hance
*                        Southern Illinois University @ Edwardsville
*			 Andrew Glassner
* 			 from "Graphics Gems", Academic Press, 1990
*
** Copyright (C) 1993 SIUE - by Gregory Hance.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
* RCS (Revision Control System) Information
* ...Added automatically by RCS 
*
** $Log: CVIPdef.h,v $
** Revision 1.1.1.1  2005/07/12 18:20:25  tingz
**
**
** Revision 1.1.1.1  2005/07/01 19:05:24  juchangh
** initialization of the new slic project
**
** Revision 1.1.1.1  2005/02/01 18:45:26  root
** SLIC
**
** Revision 1.1.1.1  2004/06/01 16:53:47  root
** adding SLIC
**
** Revision 1.1.1.1  1998/12/09 14:05:47  boland
** Imported files
**
** Revision 1.16  1997/07/11 20:47:12  smakam
** changed the definition of drand48 function so that it doesnt return a 1
**
** Revision 1.15  1997/06/16 21:32:19  yawei
** added bzero/bcopy definition for SYSV and Win32
**
** Revision 1.14  1997/05/29 20:28:55  yawei
** added strcasecmp for Win32.
**
** Revision 1.13  1997/05/29 15:23:50  yawei
** renamed NEAR definition for WIN32 compatibility
**
** Revision 1.12  1997/05/19 22:18:18  yawei
** added cbrt() macro for WIN32, used in m_sqrt().
**
** Revision 1.11  1997/05/19 22:09:21  yawei
** corrected drand48 definition
**
** Revision 1.10  1997/05/19 21:51:54  yawei
** added drand48 definition for WIN32
**
** Revision 1.9  1997/05/18 04:13:56  yawei
** added definition on 
** of ulong for WIN32
**
** Revision 1.8  1997/02/26 20:51:38  yawei
** conditionally define PI.
**
** Revision 1.7  1997/01/03 20:22:46  akjoele
** Added hooks for FreeBSD, in which ulong is not defined.
**
 * Revision 1.6  1996/12/17  20:10:48  akjoele
 * fixed NEAR definition.
 *
 * Revision 1.5  1996/12/09  19:04:52  akjoele
 * Fixed error in NEAR.  It was using ABS, which casts everything to
 * int.  Thus, no use on floats, which is the intended use of NEAR.
 *
 * Revision 1.4  1996/12/06  19:13:19  akjoele
 * Added NEAR
 *
 * Revision 1.3  1996/01/30  23:14:27  kluo
 * change ABS
 *
 * Revision 1.2  1996/01/30  23:11:30  kluo
 * change ABS
 *
 * Revision 1.1  1994/10/30  22:43:06  hanceg
 * Initial revision
 *
 * Revision 1.2  1993/05/02  23:24:01  hanceg
 * fixed comment bug
 *
 * Revision 1.1  1993/05/02  23:20:57  hanceg
 * Initial revision
 *
*
**************************************************************************/

#if !defined( CVIPDEF_DEFINED )

   #define CVIPDEF_DEFINED

   #if !defined( CVIPTYP_DEFINED )
      #include "CVIPtyp.h"
   #endif

   /**********************
   * No-argument macros *  
   **********************/

   #define             CLS   printf("\033[2J")
   #define            BEEP   printf("\7")

   /***********************/
   /* one-argument macros */
   /***********************/

   /* absolute value of a */
   #define ABS(a)		((((int)(a))<0) ? -(a) : (a))

   /* round a to nearest integer towards 0 */
   #define FLOOR(a)		((a)>0 ? (int)(a) : -(int)(-a))

   /* round a to nearest integer away from 0 */
   #define CEILING(a) \
   ((a)==(int)(a) ? (a) : (a)>0 ? 1+(int)(a) : -(1+(int)(-a)))

   /* round a to nearest int */
   #define ROUND(a)  (((a) < 0) ? (int)((a)-0.5) : (int)((a)+0.5))

   /* take sign of a, either -1, 0, or 1 */
   #define ZSGN(a)		(((a)<0) ? -1 : (a)>0 ? 1 : 0)	

   /* take binary sign of a, either -1, or 1 if >= 0 */
   #define SGN(a)		(((a)<0) ? -1 : 1)

   /* shout if something that should be true isn't */
   #define ASSERT(x) \
   if (!(x)) fprintf(stderr," Assert failed: %d\n",x);

   /* square a */
   #define SQR(a)		((a)*(a))

	#ifdef WIN32
   #define         drand48()   (((double)rand())/(RAND_MAX+1))
   #define         cbrt(a)   (pow((a),1.0/3.0))
	#endif

   /* find base 2 log of a number */
   #define LOG2(a)         log((double)(a))/log(2.0)

   /* Macro to ease the pains of e.g. comparing 1/3 with 1/3 */
   #define CVIP_NEAR(a,b)	(((float)((a)-(b))) < 0.001 && ((float)((a)-(b))) > -0.001) ? 1 : 0


   /***********************/
   /* two-argument macros */
   /***********************/

   /* find minimum of a and b */
   #define MIN(a,b)	(((a)<(b))?(a):(b))	

   /* find maximum of a and b */
   #define MAX(a,b)	(((a)>(b))?(a):(b))	

   /* swap a and b (see Gem by Wyvill) */
   /* M. Boland  #define SWAP(a,b)	{ a^=b; b^=a; a^=b; } */

   /* linear interpolation from l (when a=0) to h (when a=1)*/
   /* (equal to (a*h)+((1-a)*l) */
   #define LERP(a,l,h)	((l)+(((h)-(l))*(a)))

   /* clamp the input to the specified range */
   #define CLAMP(v,l,h)	((v)<(l) ? (l) : (v) > (h) ? (h) : v)

   /* reposition the shell cursor at line y, offset x */
   #define    GOTOYX(y, x)   printf("\033[%d;%dH", y, x)

	#ifdef WIN32
	#ifndef strcasecmp
	#define strcasecmp(a,b) stricmp(a,b)
	#endif
	#endif

	#if defined(SYSV)  || defined(WIN32)
	#ifndef bzero
	#define bzero(dst,len) memset(dst,0,len)
	#endif
	#ifndef bcopy
	#define bcopy(src,dst,len) memcpy(dst,src,len)
	#endif
	#endif


   /****************************/
   /* memory allocation macros */
   /****************************/

   /* create a new instance of a structure (see Gem by Hultquist) */
   #define NEWSTRUCT(x)	(struct x *)(malloc((unsigned)sizeof(struct x)))

   /* create a new instance of a type */
   #define NEWTYPE(x)	(x *)(malloc((unsigned)sizeof(x)))


   /********************/
   /* useful constants */
   /********************/

	#ifndef PI
   #define PI		3.141592654	/* the venerable pi */
	#endif
   #define PITIMES2	6.283185307	/* 2 * pi */
   #define PIOVER2	1.570796327	/* pi / 2 */
   #define E		2.718281828	/* the venerable e */
   #define SQRT2	1.414213562	/* sqrt(2) */
   #define SQRT2OVER2	0.7071068	/* sqrt(2) / 2 */
   #define SQRT3	1.732050808	/* sqrt(3) */
   #define GOLDEN	1.618034	/* the golden ratio */
   #define DTOR		0.017453293	/* convert degrees to radians */
   #define RTOD		57.29577951	/* convert radians to degrees */
/* For FreeBSD, ulong is not defined,
   so we'll define it here */
#if (defined(i386BSD) || defined(WIN32))
   typedef unsigned long ulong;
#endif

#endif
