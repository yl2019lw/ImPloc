function W = getBasis( idx, datadir)
% W = GETBASIS( IDX)
% inputs:   IDX is the antibody id for a particular protein (default 1873)
% outputs:  W is the color basis matrix
% 
% Saves W to ./lib/2_separationCode.

if ~exist('datadir','var')
	datadir = './data/1_images';
end

tissuedir = [datadir '/' num2str(idx) '/normal'];

if ~exist( tissuedir,'dir')
    error('no antibody id');
end

d2 = dir(tissuedir);
d2(1:2) = [];

count = 1;
for j=1:length(d2)
    disp(j);
    imgdir = [tissuedir '/' d2(j).name];

    d3 = dir(imgdir);
    d3(1:2) = [];
    


    for k=1:length(d3)
        infile = [imgdir '/' d3(k).name];
        I = imread( infile);
        counter = 1;
        eval('[W] = colorbasis( I);','counter = 0;');
        if counter
            Wbasis{count} = W;
            count = count + 1;
        end
    end
end


W = zeros(size(Wbasis{1}));
for i=1:length(Wbasis)
    W = W + Wbasis{i};
end
W=W';

return



function W = colorbasis( I, STOPCONN, A_THR, S_THR)

rank = 2;  ITER = 5000;

tic;

if ~exist('STOPCONN','var')
    STOPCONN = 40;
end
if ~exist('A_THR','var')
    A_THR = 1000;
end
if ~exist('S_THR','var')
    S_THR = 1000;
end

I = 255 - I;
IMGSIZE = size(I);

% ....tissue size check!
if (IMGSIZE(1)<S_THR) || (IMGSIZE(2)<S_THR)
	error('Not enough useable tissue staining');
end


% ********** SEED COLORS ************
S = size(I);
V = reshape( I, S(1)*S(2),S(3));
[V ind VIDX] = unique(V,'rows');
VIDX = single(VIDX);

HSV = rgb2hsv( V);
hue = HSV(:,1);
[c b] = hist( hue(hue<0.3), [0:0.01:1]);
[A i] = max(c);
P = b(i);
hae = mean(V(P-.01<hue & hue<P+0.01,:),1)';

[c b] = hist( hue(hue>=0.3), [0:0.01:1]);
[A i] = max(c);
P = b(i);
dab = mean(V(P-.01<hue & hue<P+0.01,:),1)';

W = single( [hae dab] / 255);

