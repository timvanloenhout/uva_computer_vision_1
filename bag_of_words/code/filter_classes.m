function [X_filtered, y_filtered] = filter_classes(X, y, all_classes, keep_classes)
    

class_idxs = 1:size(all_classes, 2);
used_classes = contains(all_classes, keep_classes);
indexes = ismember(y, used_classes .* class_idxs);

X_filtered = X(indexes,:,:,:);
y_filtered = y(indexes, :);

end
