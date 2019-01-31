function H = findH( V, W)
% H = FINDH( V, W) linearly unmixes a set of N m-dimensional samples (Nxm matrix)
%   inputs: V, the original samples
%           W the color basis matrix, m-dimensioanl, r=rank (mxr)
%
%   outputs: H, the unmixed weight channels

H = single(V)*pinv(single(W));
H = H - min(H(:));
H = H / max(H(:))*255;
H = uint8(round(H));
