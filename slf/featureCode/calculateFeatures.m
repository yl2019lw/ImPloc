function calculateFeatures(readpath, writepath, dbtype, NLEVELS, feattype)
% CALCULATEFEATURES( READPATH, WRITEPATH, DBTYPE, NLEVELS)
% disp(['calc readpath---->', readpath]);
% disp(['calc writepath---->', writepath]);
% disp(['calc dbtype---->', dbtype]);
% disp(['calc nlevels---->', num2str(NLEVELS)]);
% disp(['calc feattype---->', feattype]);

if strcmp(feattype, 'SLFs')
    if exist( writepath,'file')
        load (writepath);
        if length(features)==841
            return
        end
    end
    prp = writepath;
       prp((prp == '/') | (prp == '.')) = '_';
       pr = ['tmp/calculating_' prp '.txt'];
      if ~exist( pr,'file')
        fr = fopen( pr, 'w');
      else
        return
      end
      
    I = imfinfo( readpath);
    H = imread( readpath);
    J = reconIH( imread(I.Comment), H);

    features = tissueFeatures( J(:,:,2), J(:,:,1), dbtype, NLEVELS, feattype);

    save( writepath, 'features');

    fclose(fr);
    delete(pr);
end

if strcmp(feattype, 'SLFs_LBPs')
%    if exist( writepath,'file')
%         load (writepath);
%         if length(features)==1097
%             return
%         end
%    end
    prp = writepath;
       prp((prp == '/') | (prp == '.')) = '_';
       pr = ['tmp/calculating_' prp '.txt'];
      if ~exist( pr,'file')
        fr = fopen( pr, 'w');
      else
        return
      end
      
    I = imfinfo( readpath);
    H = imread( readpath);
    J = reconIH( imread(I.Comment), H);

    features = tissueFeatures( J(:,:,2), J(:,:,1), dbtype, NLEVELS, feattype);

    save( writepath, 'features');
    fclose(fr);
    delete(pr);
end 
 if strcmp(feattype, 'SLFs_LBPs_CL')
   if exist( writepath,'file')
        load (writepath);
        if length(features)==1097
            return
        end
   end
    prp = writepath;
       prp((prp == '/') | (prp == '.')) = '_';
       pr = ['tmp/calculating_' prp '.txt'];
      if ~exist( pr,'file')
        fr = fopen( pr, 'w');
      else
        return
      end
      
    I = imfinfo( readpath);
    H = imread( readpath);
    J = reconIH( imread(I.Comment), H);
    J(:,:,1) = adapthisteq(J(:,:,1));
     J(:,:,2) = adapthisteq(J(:,:,2));
    features = tissueFeatures( J(:,:,2), J(:,:,1), dbtype, NLEVELS, feattype);

    save( writepath, 'features');
    fclose(fr);
    delete(pr);
 end 
end
    
 