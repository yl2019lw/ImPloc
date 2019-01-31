/***************************************************************************
* ======================================================================
* Computer Vision/Image Processing Tool Project - Dr. Scott Umbaugh SIUE
* ======================================================================
*
*             File Name: CVIPtyp.h
*           Description: defines many of the standard types used throughout
*			 CVIPtools source files
*         Related Files: most if not all CVIPtools source files include
*			 this header 
*   Initial Coding Date: 1/3/93
*           Portability: Standard (ANSI) C
*             Credit(s): Gregory Hance
*                        Southern Illinois University @ Edwardsville
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
** $Log: CVIPtyp.h,v $
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
 * Revision 1.3  1997/03/08  17:02:11  yawei
 * Swaped CVIP_YES and CVIP_NO
 *
** Revision 1.2  1997/03/08 00:47:13  yawei
** Name Changes:
** 	BOOLEAN ==> CVIP_BOOLEAN
** 	FALSE ==> CVIP_YES
** 	TRUE ==> CVIP_NO
** 	BYTE ==> CVIP_BYTE
** 	SHORT ==> CVIP_SHORT
** 	INTEGER ==> CVIP_INTEGER
** 	FLOAT ==> CVIP_FLOAT
** 	DOUBLE ==> CVIP_DOUBLE
** 	TYPE ==> CVIP_TYPE
**
** Revision 1.1  1994/10/30 22:43:06  hanceg
** Initial revision
**
 * Revision 1.1  1993/05/02  23:21:01  hanceg
 * Initial revision
 *
*
****************************************************************************/

#if !defined( CVIPTYPE_DEFINED )

   #define CVIPTYPE_DEFINED

   typedef enum {CVIP_NO, CVIP_YES} CVIP_BOOLEAN;
   typedef enum {OFF, ON} STATE;
   typedef enum {CVIP_BYTE, CVIP_SHORT, CVIP_INTEGER, CVIP_FLOAT, CVIP_DOUBLE} CVIP_TYPE;

   typedef unsigned char byte;

#endif
