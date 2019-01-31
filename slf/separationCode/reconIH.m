function [J,ma] = reconIH( I,H,IDX,sc)
% J = RECONIH( I, H) reconstruct an NMF unmixed image
%   Takes as input I, the image to be unmixed, and H, the sorted pixel-weight channels.
%   Returns J, the unmixed image. Should be called after findWH.m

if ~exist( 'IDX','var')
    IDX = [];
end
if ~exist( 'sc','var')
    sc = 0;
end

ma = [];

s = size(I);

if ~length(IDX)
    [u i IDX] = unique( 255-reshape( I, [s(1)*s(2) s(3)]), 'rows');
end

J = H(IDX,:);
if sc
    ma = zeros(size(H,2),1);
    for i=1:size(H,2)
        [c b] = imhist(J(:,i));
        [a ind] = max(c);
        J(:,i) = J(:,i)-b(ind);
        ma(i) = max(J(:,i));
    end
    J = (255/max(J(:)))*J;
end

J = reshape( J, [s(1) s(2) size(H,2)]);

if size(H,2)<3
    J(:,:,3) = 0;
end

