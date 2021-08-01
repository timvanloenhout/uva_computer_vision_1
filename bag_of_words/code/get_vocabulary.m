function [vocab] = get_vocabulary(data, colorspace, sampling_type, vocab_size)

all_descriptors = [];

fprintf('Getting the descriptors of the input images...\n')

for i=1:size(data, 1)
    img_desc = get_descriptors(squeeze(data(i,:,:,:)), colorspace, sampling_type);
    all_descriptors = [all_descriptors; img_desc];
end

fprintf('Clustering the descriptors...\n')

[~, vocab] = kmeans(double(all_descriptors), vocab_size);