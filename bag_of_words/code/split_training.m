function [y_SVM, X_SVM, X_voc] = split_training(X, y, ratio)

    fprintf('Splitting the datasets...\n')

    % Create zero matrices (for later concatination)
    SVM_size = round(size(X, 1)*ratio);
    voc_size = round(size(X, 1)*(1-ratio));
    
    y_SVM = zeros(size(y(1:1,:,:,:)));
    X_SVM = zeros(size(X(1:1,:,:,:)));
    X_voc = zeros(size(X(1:1,:,:,:)));
    
    % Fill zero matrices
    uniques = unique(y);
    class_size_SVM = SVM_size / size(uniques,1);
    class_size_voc = voc_size / size(uniques,1);
    for i = 1:size(uniques)
        % Get indices for class i
        j = uniques(i);
        class_indices = find(y(:,1) == j);
        % Split indices into indices for SVM and vocabulary
        class_indices_SVM = class_indices(1:class_size_SVM);
        class_indices_voc = class_indices(class_size_SVM+1:size(class_indices)); 
        % Add class i entries to the correct matrices
        yi_SVM = y(class_indices_SVM,:,:,:);
        y_SVM = vertcat(y_SVM, yi_SVM);
        Xi_SVM = X(class_indices_SVM,:,:,:);
        X_SVM = vertcat(X_SVM, Xi_SVM);
        Xi_voc = X(class_indices_voc,:,:,:);
        X_voc = vertcat(X_voc, Xi_voc);       
    end

    y_SVM = y_SVM(2:size(y_SVM,1),:,:,:);
    X_SVM = X_SVM(2:size(X_SVM,1),:,:,:);
    X_voc = X_voc(2:size(X_voc,1),:,:,:);
    
    shuffle_SVM = randperm(length(X_SVM));
    y_SVM = y_SVM(shuffle_SVM,:,:,:);
    X_SVM = X_SVM(shuffle_SVM,:,:,:);
    
    shuffle_voc = randperm(length(X_voc));
    X_voc = X_voc(shuffle_voc,:,:,:);
    
end