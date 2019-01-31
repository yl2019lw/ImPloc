/* pbm.h - header file for libpbm portable bitmap library
 * @(#)pbm.h	1.1 12/18/92
 */

#ifndef _PBM_H_
#define _PBM_H_

#include "ppbmplus.h"

typedef unsigned char bit;
#define PBM_WHITE 0
#define PBM_BLACK 1

/* Declarations of routines. */

#define pbm_allocarray( cols, rows ) ((bit **) pm_allocarray( cols, rows, sizeof(bit) ))
#define pbm_allocrow( cols ) ((bit *) pm_allocrow( cols, sizeof(bit) ))
#define pbm_freearray( bitrow, rows ) pm_freearray( bitrow, rows )
#define pbm_freerow( bitrow ) pm_freerow( bitrow )

bit **pbm_readpbm( /* FILE *file, int *colsP, int *rowsP */ );
void pbm_readpbminit( /* FILE *file, int *colsP, int *rowsP, int *formatP */ );
void pbm_readpbmrow( /* FILE *file, bit *bitrow, int cols, int format */ );

void pbm_writepbm( /* FILE *file, bit **bits, int cols, int rows */ );
void pbm_writepbminit( /* FILE *file, int cols, int rows */ );
void pbm_writepbmrow( /* FILE *file, bit *bitrow, int cols */ );

#endif /*_PBM_H_*/
