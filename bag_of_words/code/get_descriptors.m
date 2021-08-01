function [descriptors] = get_descriptors(image, colorspace, sampling_type)

if size(image,3) == 3
    gray_image = single(rgb2gray(image));
else 
    gray_image = single(image);
end

% Choose colorspace for SIFT
switch lower(colorspace)
    case "grayscale"
        image = gray_image;
    case "rgb"
        image = single(image);      
    case "opponent"
        image = single(rgb2opponent(image));
    otherwise
        error("Colorspace not recognized");
end

sampling_type = lower(sampling_type);
descriptors = [];

% Smooth image for dense sampling
% if strcmp(sampling_type, 'dense')
%     image = vl_imsmooth();

% Choose dense or keypoint sampling
    if strcmp(sampling_type, 'dense')
        for chan = 1:size(image, 3)
            [~, tmp_desc] = vl_dsift(image(:,:,chan), 'Step', 5, 'Size', 21);
            descriptors = [descriptors; tmp_desc'];
        end
    elseif strcmp(sampling_type,"key_point")
        for chan = 1:size(image, 3)
            [~, tmp_desc] = vl_sift(image(:,:,chan));
            descriptors = [descriptors; tmp_desc'];
        end
%         if size(image, 3) == 3
%             tmp_img = single(rgb2gray(image));
%         else
%             tmp_img = image;
%         end
%         f = vl_sift(tmp_img);
%         for chan = 1:size(image, 3)
%             [~, tmp_desc] = vl_covdet(image(:,:,chan), 'Frames', f, 'descriptor', 'sift');
%             descriptors = [descriptors; tmp_desc'];
%         end
    else
        error("Sampling not recognized");
    end
end
