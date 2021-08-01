function [X, y, class_names] = load_dataset(path, img_dim)
    
fprintf("Loading %s...\n", path);

dataset = load(path);
class_names = dataset.class_names;
X = dataset.X;
y = dataset.y;

ds_size = size(X, 1);
X = reshape(X, [ds_size img_dim]);
end
