
INPUT_DIR = 'separation_data';
OUTPUT_DIR = 'feature_data';
dbs = {'db1','db2','db3','db4','db5','db6','db7','db8','db9','db10'};

GENES = dir(INPUT_DIR);
% remove '.' and '..' from outputs of dir
GENES(1:2) = [];


options.NLEVELS = 10;
options.FEATTYPE = 'SLFs_LBPs';

addpath separationCode;
addpath featureCode;

matlabpool open 4
for gi =1:length(GENES)
    if mod(gi, 2) == 1
        continue
    end
    gene = GENES(gi).name;
    gene_dir = [INPUT_DIR '/' gene];
    out_gene_dir = [OUTPUT_DIR '/' gene];
    if ~exist(out_gene_dir, 'dir')
        mkdir(out_gene_dir);
    end
    gene_imgs = dir(gene_dir);
    gene_imgs(1:2) = [];
    parfor imgi = 1:length(gene_imgs)
        
        img = gene_imgs(imgi).name;
        inputpath = [gene_dir '/' img];
        for dbi = 1:length(dbs)
            cdb = dbs(dbi);
            out = [img(1:end-4), '-', cdb, '.mat'];
            out = [out{:}];
   
            outpath = [out_gene_dir '/' out];
            if exist(outpath, 'file')
                disp(['already calc feature for outputpath---->', outpath]);
                continue;
            end
            disp(['inputpath---->', inputpath]);
            disp(['outputpath---->', outpath]);
            calculateFeatures(inputpath, outpath, char(cdb), options.NLEVELS, options.FEATTYPE);
        end
    end
end 

matlabpool close