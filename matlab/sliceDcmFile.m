function y = sliceDcmFile(file, output)
    [y,~,~] = dicomread(file.name);
    
    y = mat2gray(y);
    [~,name,~] = fileparts(file.name);
    imwrite(y, [fullfile(output, name) '.jpg']);
end