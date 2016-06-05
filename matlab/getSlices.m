function y = getSlices(dir)
    outputFolder = 'C:\results\';
    sliceNo = 15;
    
    if ~isdir(outputFolder)
        errorMessage = sprintf('Error: Please create output folder:\n%s', outputFolder);
        uiwait(warndlg(errorMessage));
        return;
    end
    
    %niiFiles = rdir([dir, '\**\*.nii']);
    %for i = 1:numel(niiFiles);
        %disp(files(i));
    %    sliceNiiFile(sliceNo, niiFiles(i), outputFolder);
    %end
    
    name = sprintf('\\**\\*%d.dcm', sliceNo);
    dcmFiles = rdir([dir, name]);
    for j = 1:numel(dcmFiles);
        disp(dcmFiles(j));
        sliceDcmFile(dcmFiles(j), outputFolder);
    end

    y = 'OK';
end