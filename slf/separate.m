
INPUT_DIR = 'enhance_4tissue_data';
OUTPUT_DIR = 'separation_data';

GENES = dir(INPUT_DIR);
% remove '.' and '..' from outputs of dir
GENES(1:2) = [];
count = 0;

for gi =1:length(GENES)
    gene = GENES(gi).name;
    gene_dir = [INPUT_DIR '/' gene];
    out_gene_dir = [OUTPUT_DIR '/' gene];
    if ~exist(out_gene_dir, 'dir')
        mkdir(out_gene_dir);
    end
    gene_imgs = dir(gene_dir);
    gene_imgs(1:2) = [];
    for imgi = 1:length(gene_imgs)
        count = count + 1;
        img = gene_imgs(imgi).name;
        input_imgs{count} = [gene_dir '/' img];
        
        outimg = [img(1:end-3) 'png'];
        output_imgs{count} = [out_gene_dir '/' outimg];

    end
end 

% for i=1:count
%     disp(input_imgs{i});
%     disp(output_imgs{i})
% end

addpath separationCode;

options.UMETHOD = 'lin';
options.ITER = 5000;
options.INIT = 'truncated';
options.RSEED = 13;
options.RANK = 2;
options.STOPCONN = 40;
options.VERBOSE = 1;
bf = 'Wbasis.mat';
load( bf);
options.W = W;

matlabpool open 4
parfor index = 1:count
    readpath = input_imgs{index};
    writepath = output_imgs{index};
    disp(['readpath----> ', readpath]);
    disp(['writepath----> ', writepath]);
    processImage( readpath, writepath, options);
end
matlabpool close
