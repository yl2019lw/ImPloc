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
/*/////////////////////////////////////////////////////////////////////////
//
//
//                            ml_texture.c 
//
//
//                           Michael Boland
//                            23 Nov 1998
//
//  Revisions: Name Change - EJSR
//
/////////////////////////////////////////////////////////////////////////*/


#include "mex.h"
#include "matrix.h"
#include "ppgm.h"
#include "CVIPtexture.h"
#include <sys/types.h>

#define row 0
#define col 1

extern TEXTURE * Extract_Texture_Features();

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{

  int         distance;                 /*parameter for texture calculations*/
  unsigned char*   p_img;                    /*The image from Matlab*/
  unsigned char**  p_gray;                   /*Image converted for texture calcs*/
  int         mrows;                    /*Image height*/
  int         ncols;                    /*Image width*/
  TEXTURE_FEATURE_MAP* features_used ;  /*Indicate which features to calc.*/
  TEXTURE*    features ;                 /*Returned struct of features*/
  int         i ;
  int         imgsize[2] ;              
  int         imgindex[2] ;
  long        offset ;                  
  int         outputsize[2] ;           /*Dimensions of TEXTURE struct*/
  int         outputindex[2] ;
  float*      output ;                  /*Features to return*/
  unsigned char*   outtest ;

  if (nrhs != 1) {   //输入参数个数
    mexErrMsgTxt("ml_texture requires a single input argument.\n") ;
  } else if (nlhs != 1) {  //输出参数个数
    mexErrMsgTxt("ml_texture returns a single output.\n") ;
  }

  if (!mxIsNumeric(prhs[0])) {
    mexErrMsgTxt("ml_texture requires a single numeric input.\n") ;
  }

  if (!mxIsUint8(prhs[0])) {
    mexErrMsgTxt("ml_texture requires a single input of type unsigned 8-bit integer.\n") ;
  }

  mrows = mxGetM(prhs[0]) ;
  ncols = mxGetN(prhs[0]) ;

  if(!(mrows > 1) || !(ncols > 1)) {
    mexErrMsgTxt("ml_texture requires an input image, not a scalar.\n") ;
  }

  p_img = (unsigned char*)mxGetData(prhs[0]) ;

  distance = 1 ;

  features_used = mxCalloc(1, sizeof(TEXTURE_FEATURE_MAP)) ;
  if(!features_used) 
    mexErrMsgTxt("ml_texture: error allocating features_used.") ;

  features_used->ASM = 1 ;
  features_used->contrast = 1 ;
  features_used->correlation = 1 ;
  features_used->variance = 1 ;
  features_used->IDM = 1 ;
  features_used->sum_avg = 1 ;
  features_used->sum_var = 1 ;
  features_used->sum_entropy = 1 ;
  features_used->entropy = 1 ;
  features_used->diff_var = 1 ;
  features_used->diff_entropy = 1 ;
  features_used->meas_corr1 = 1 ;
  features_used->meas_corr2 = 1 ;
  features_used->max_corr_coef = 0 ;

  imgsize[col] = ncols ;
  imgsize[row] = mrows ;
  
  p_gray = mxCalloc(mrows, sizeof(unsigned char*)) ;
  if(p_gray) {
    for(i=0; i<mrows ; i++) {
      p_gray[i] = mxCalloc(ncols, sizeof(unsigned char)) ;
      if(!p_gray[i]) mexErrMsgTxt("ml_texture : error allocating p_gray[i]") ;
    }
  } else mexErrMsgTxt("ml_texture : error allocating p_gray") ;
  

  for(imgindex[row] = 0 ; imgindex[row] < imgsize[row] ; imgindex[row]++) 
    for(imgindex[col] = 0 ; imgindex[col] < imgsize[col] ; imgindex[col]++ ) {
      offset = mxCalcSingleSubscript(prhs[0], 2, imgindex) ;
      p_gray[imgindex[row]][imgindex[col]] = p_img[offset] ;
    }

  features=Extract_Texture_Features(distance,p_gray,mrows,ncols,features_used) ;

  /*
  outputsize[row] = mrows ;
  outputsize[col] = ncols ;
  plhs[0] = mxCreateNumericArray(2, outputsize, mxUINT8_CLASS, mxREAL) ;
  if (!plhs[0]) mexErrMsgTxt("ml_texture: error allocating return variable.") ;
  outtest = (unsigned char*)mxGetData(plhs[0]) ;
  for(imgindex[row] = 0 ; imgindex[row] < imgsize[row] ; imgindex[row]++) 
    for(imgindex[col] = 0 ; imgindex[col] < imgsize[col] ; imgindex[col]++ ) {
      offset = mxCalcSingleSubscript(prhs[0], 2, imgindex) ;
      outtest[offset] = p_gray[imgindex[row]][imgindex[col]] ;
    }
  */


  outputsize[row] = 14 ;
  outputsize[col] = 6 ;
 
  plhs[0] = mxCreateNumericArray(2, outputsize, mxSINGLE_CLASS, mxREAL) ;
  if (!plhs[0]) mexErrMsgTxt("ml_texture: error allocating return variable.") ;

  output = (float*)mxGetData(plhs[0]) ;

  /* Copy the features into the return variable */

  for(outputindex[col]=0 ; outputindex[col] < outputsize[col] ; 
      outputindex[col]++) {
    outputindex[row]=0 ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->ASM[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->contrast[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->correlation[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->variance[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->IDM[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->sum_avg[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->sum_var[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->sum_entropy[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->entropy[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->diff_var[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->diff_entropy[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->meas_corr1[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->meas_corr2[outputindex[col]] ;

    outputindex[row]++ ;
    offset =  mxCalcSingleSubscript(plhs[0], 2, outputindex) ;
    output[offset] = features->max_corr_coef[outputindex[col]] ;
  }

    

  /*
    Memory clean-up.
  */
  for(i=0; i<mrows ; i++) {
    mxFree(p_gray[i]) ;
  }
  mxFree(p_gray) ;  
  mxFree(features_used) ;
  /* val is calloc'd inside of Extract_Texture_Features */
  free(features) ;
  
}
