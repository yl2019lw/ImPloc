function hfeatures = ml_texture(I)
% ML_TEXTURE(I) Haralick texture features for image I
% V = ML_TEXTURE(I),
%     Returns an array of texture features for image I.
%     The value for each of the following 13 statistics is
%     included for each of the four directions.  The mean and range of
%     the features over these four directions are also included 
%     (i.e. for each feature -- 0 deg, 45 deg, 90 deg, 135 deg, mean, range)
%
%     1) Angular second moment
%     2) Contrast
%     3) Correlation
%     4) Sum of squares
%     5) Inverse difference moment
%     6) Sum average
%     7) Sum variance
%     8) Sum entropy
%     9) Entropy
%    10) Difference variance
%    11) Difference entropy
%    12) Information measure of correlation 1
%    13) Information measure of correlation 2
%    14) Maximal correlation coefficient - NOT calculated.
