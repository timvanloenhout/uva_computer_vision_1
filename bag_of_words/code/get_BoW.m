function [BoW] = get_BoW(images, vocab, colorspace, sampling_type)

fprintf('Creating Bag of Words...\n')

vocab_size = size(vocab, 1);
data_size = size(images, 1);
BoW = zeros(data_size, vocab_size);

for i=1:data_size
    img = squeeze(images(i,:,:,:));
    
    % get descriptors and find the closest ones from the vocabulary
    descriptors = get_descriptors(img, colorspace, sampling_type);
    nearest_idxs = knnsearch(vocab, double(descriptors));
    values = unique(nearest_idxs);
    count = histc(nearest_idxs, values);
    
    his = zeros(vocab_size,  1);
    his(values) = count;
    BoW(i,:) = his/sum(his);
end
end