% This make.m is used for Extracting Haralick Features.

mex -O -c ml_texture.c
mex -O -c cvip_pgmtexture.c

mex -O ml_texture.obj cvip_pgmtexture.obj
% mex -O ml_texture.c ml_texture.obj
% mex -O -c cvip_pgmtexture.c cvip_pgmtexture.obj