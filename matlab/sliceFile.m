function y = sliceFile(sliceNo, file)
    myFolder = 'C:\results\';
    if ~isdir(myFolder)
        errorMessage = sprintf('Error: Please create output folder:\n%s', myFolder);
        uiwait(warndlg(errorMessage));
        return;
    end
    nii = load_nii(file.name);
    slice = squeeze(nii.img(:,:,sliceNo));
    tmpSlice = imresize(slice,[256,256]);

    y = mat2gray(tmpSlice);
    [~,name,~] = fileparts(file.name);
    disp(name);
    disp(fullfile(myFolder, name));
    imwrite(y, [fullfile(myFolder, name) '.jpg']);
end