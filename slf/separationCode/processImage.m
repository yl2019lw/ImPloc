function processImage( readpath, writepath, params)
% PROCESSIMAGE( READPATH, WRITEPATH, PARAMS)

if exist( writepath,'file')
    return
end

prp = writepath;
prp((prp == '/') | (prp == '.')) = '_';
pr = ['tmp/processing_' prp '.txt'];
if ~exist( pr,'file')
    fr = fopen( pr, 'w');
else
    return
end

% Always perform linear unmixing...
I = imread( readpath);
W = params.W;

s = size(I);
[V i j] = unique( 255-reshape( I, [s(1)*s(2) s(3)]), 'rows');
V = unique(V,'rows');

% LIN
H = findH( V, W);

% ...Note images without enough staining
% J = reconIH( I, H, j);
% [c b] = imhist(J(:,:,1));
% [a ind] = max(c);
% J(:,:,1) = J(:,:,1) - b(ind);
% [c b2] = imhist(J(:,:,2));
% [a ind2] = max(c);
% J(:,:,2) = J(:,:,2) - b2(ind2);
% ratio = sum(sum(J(:,:,2))) / sum(sum(J(:,:,1)));
% 
% if ratio<.5
%     imwrite( 0, writepath, 'comment', readpath);
%     fclose(fr);
%     delete(pr);
%     return
% end

if strcmp( params.UMETHOD,'nmf')
    [W,H] = findWH( V, params);
    % J = reconIH( I, H, j);
end

imwrite( H, writepath, 'comment', readpath);

fclose(fr);
delete(pr);

return
