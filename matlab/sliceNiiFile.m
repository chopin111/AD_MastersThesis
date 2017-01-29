function y = sliceNiiFile(sliceNo, file, output)

    try
        nii = load_nii(file.name);
    catch
        nii = load_untouch_nii(file.name);
    end
    [a,b,c] = size(nii.img);
    
    if c < 256
        sliceNo = sliceNo - 20;
    end
    
    for i = 0:150;
        slice = squeeze(nii.img(:,:,c-i));
        tmpSlice = imresize(slice,[256,256]);
        if (max(max(tmpSlice)) > 1000)
           break;
        end
    end
    if i == 150 || c-i-sliceNo <= 0
        fprintf('Incorrect file %s', file.name);
        return;
    end
    
    fprintf('Starting slice no: %d',i);
    slice = squeeze(nii.img(:,:,c-i-sliceNo));
    tmpSlice = imresize(slice,[256,256]);
        
    y = mat2gray(tmpSlice);
    [~,name,~] = fileparts(file.name);
    imwrite(y, [fullfile(output, name) '.jpg']);
end