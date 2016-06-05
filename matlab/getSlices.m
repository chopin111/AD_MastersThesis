function y = getSlices(dir)
    files = rdir([dir, '\**\*.nii']);

    for i = 1:numel(files);
        disp(files(1));
        y = sliceFile(180, files(i));
    end
end