nii = load_nii('test1.nii');
sliceNo = 2;
slice = squeeze(nii.img(:,:,sliceNo));
tmpSlice = imresize(slice,[256,256]);

i = mat2gray(tmpSlice);

imshow(i);