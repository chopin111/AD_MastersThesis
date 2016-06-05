function y = sliceNiiFile(sliceNo, file, output)
    nii = load_nii(file.name);
    slice = squeeze(nii.img(:,:,sliceNo));
    tmpSlice = imresize(slice,[256,256]);

    y = mat2gray(tmpSlice);
    [~,name,~] = fileparts(file.name);
    %disp(name);
    %disp(fullfile(myFolder, name));
    imwrite(y, [fullfile(output, name) '.jpg']);
end