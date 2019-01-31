function wfeat = tissueFeatures( prot, dna, dbtype, NLEVELS, feattype)
% TISSUEFEATURES   Extract tissue features from unmixed image
%    FEAT = TISSUEFEATURES( prot, dna, DBTYPE, NLEVELS) 
%    calculates 839 morphological and wavelet texture features.
%    I is the protein image, J is the nuclear image, DBTYPE
%    is the type of filter that should be used for the wavelet
%    decomposition, and NLEVELS is the number of decomposition 
%    levels.
% 
%    Inputs values are: 
%    prot = graylevel, uint8 protein image
%    dna = graylevel, uint8 nuclear image
%    DBTYPE = 'db1',db2',..., or 'db10'
%    NLEVELS = 1,2..., whatever is allowed by image dimensions.
%         A value of 10 is typically used

% BG subtraction by most-common-pixel
[c b] = imhist(prot);
[a i] = max(c);
prot = prot - b(i);

[c b] = imhist(dna);
[a i] = max(c);
dna = dna - b(i);

% Threshold by Otsu's method
thr_prot = graythresh(prot)*255;
thr_dna = graythresh(dna)*255;

obj_prot = prot>thr_prot;
obj_dna = dna>thr_dna;

S = size(prot);
% Feat1: ratio of areas
area_prot = sum(obj_prot(:));
area_dna = sum(obj_dna(:));
% rel_area_prot=area_prot / (S(1)*S(2));
SLFareaRatio = area_prot / area_dna;

% p1 = prot - thr_prot;
% E = edge( p1, 'canny');
% efeat1 = sum(E(:)) / area_prot; % ratio above threshold pixels along edge to above threshold pixels
% efeat2 = sum( p1(E(:))) / sum( p1(:)); % ratio fluorescence along edge to total fluorescence

overlap = obj_prot & obj_dna;
areaOverlap = sum(overlap(:));
SLFoverlapRatio = areaOverlap / area_prot;

SLFoverlapIntRatio = sum(prot(overlap)) / sum(prot(:));

dist_dna = bwdist(obj_dna);
SLFdistance = sum(dist_dna(obj_prot))/area_prot;

% ofeat = [rel_area_prot SLFareaRatio SLFoverlapRatio SLFoverlapIntRatio SLFdistance efeat1 efeat2];
ofeat = [area_prot SLFareaRatio SLFoverlapRatio SLFoverlapIntRatio SLFdistance];
clear dist_* obj_ overlap dna*

prot = single(prot);

%¼ÆËãproteinÍ¼ÏñµÄlbpÌØÕ÷
if strcmp(feattype,'SLFs_LBPs')||strcmp(feattype,'SLFs_LBPs_CL')
   spoints = [1, 0; 1, -1; 0, -1; -1, -1; -1, 0; -1, 1; 0, 1; 1, 1];
   lbpfeat=lbp(prot,spoints,0,'h');
end

GLEVELS = 31;
A = uint8(round(GLEVELS*prot/max(prot(:))));

wfeat = ml_texture( A);
wfeat = [mean(wfeat(1:13,[1 3]),2); mean(wfeat(1:13,[2 4]),2)]';
% disp(['NLEVEls', num2str(NLEVELS)]);
% disp(['dbtype', dbtype]);

[C,S] = wavedec2(prot,NLEVELS,dbtype);

for k = 0 : NLEVELS-1
    [chd,cvd,cdd] = detcoef2('all',C,S,(NLEVELS-k));
    A = chd - min(chd(:));
    A = uint8(round(GLEVELS*A/max(A(:))));
    hfeat = ml_texture( A);
    hfeat = [mean(hfeat(1:13,[1 3]),2); mean(hfeat(1:13,[2 4]),2)]';

    A = cvd - min(cvd(:));
    A = uint8(round(GLEVELS*A/max(A(:))));
    vfeat = ml_texture( A);
    vfeat = [mean(vfeat(1:13,[1 3]),2); mean(vfeat(1:13,[2 4]),2)]';

    A = cdd - min(cdd(:));
    A = uint8(round(GLEVELS*A/max(A(:))));
    dfeat = ml_texture( A);
    dfeat = [mean(dfeat(1:13,[1 3]),2); mean(dfeat(1:13,[2 4]),2)]';

    wfeat = [wfeat hfeat vfeat dfeat ...
        sqrt(sum(sum(chd.^2))) ...
        sqrt(sum(sum(cvd.^2))) ...
        sqrt(sum(sum(cdd.^2)))];
end


if strcmp(feattype,'SLFs')
   wfeat = [ ofeat wfeat];
end

if strcmp(feattype,'SLFs_LBPs')||strcmp(feattype,'SLFs_LBPs_CL')
   wfeat = [ ofeat wfeat lbpfeat];
end


return

