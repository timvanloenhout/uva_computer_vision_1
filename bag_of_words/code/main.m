tic

% PARAMETERS
keep_classes = {'airplane','bird','ship','horse','car'};
img_shape = [96,96,3];
number_of_top_and_bottom_images= 5;

% HYPERPARAMETERS
split_ratios = [0.9, 0.8];
sampling_types = ["key_point", "dense"];
vocab_sizes = [400, 1000, 4000];
colorspaces = ["grayscale", "rgb", "opponent"];

[Xs_train, ys_train, class_names] = load_dataset('train.mat', img_shape);
[Xs_train, ys_train] = filter_classes(Xs_train, ys_train, class_names, keep_classes);

[X_test, y_test, class_names] = load_dataset('test.mat', img_shape);
[X_test, y_test] = filter_classes(X_test, y_test, class_names, keep_classes);

sampling_type_bar_chart_values = cell(size(sampling_types, 2), 1);
vocab_size_bar_chart_values = cell(size(vocab_sizes, 2), 1);
colorspaces_bar_chart_values = cell(size(colorspaces, 2), 1);
split_ratio_bar_chart_values = cell(size(split_ratios, 2), 1);

sampling_type_bar_legend = cell(size(sampling_types, 2), 1);
vocab_size_bar_legend = cell(size(vocab_sizes, 2), 1);
colorspaces_bar_legend = cell(size(colorspaces, 2), 1);
split_ratio_bar_legend = cell(size(split_ratios, 2), 1);

for sr=1:length(split_ratios)
    split_ratio = split_ratios(sr);
    
    for s=1:length(sampling_types)
        sampling_type = sampling_types(s);
        
        for c=1:length(colorspaces)
            colorspace = colorspaces(c);

            for vc=1:length(vocab_sizes)
                vocab_size = vocab_sizes(vc);

                % Divide data into set for get_vocabulary and a set for training the SVM
                [y_train, X_train, X_train_voc] = split_training(Xs_train, ys_train, split_ratio);

                vocab = get_vocabulary(X_train_voc, colorspace, sampling_type, vocab_size);

                BoW_train = get_BoW(X_train, vocab, colorspace, sampling_type);
                BoW_test = get_BoW(X_test, vocab, colorspace, sampling_type);

                unique_classes = unique(y_train);
                unique_classes_size = size(unique_classes);

                average_precisions_all = zeros(unique_classes_size);

                for i = 1:unique_classes_size

                    % Remap the labels to binary labels 
                    %(1 for the current class and 0 for the rest)
                    curr_class = unique_classes(i);
                    model_ys = y_train == curr_class;

                    fprintf('Training for %s class \n', class_names{curr_class})

                    model = fitcsvm(BoW_train, model_ys, 'KernelFunction', 'Gaussian');

                    [predictions, confidence_scores] = predict(model, BoW_test);
                    [confidence_scores, sorting_indexes] = sort(confidence_scores(:, 1));
                    predictions = predictions(sorting_indexes);

                    % Sort the test dataset according to the sorting order of the
                    % predictions
                    sorted_x_test = X_test(sorting_indexes, :,:,:);
                    sorted_y_test = y_test(sorting_indexes);

                    % Remap the labels to binary labels 
                    %(1 for the current class and 0 for the rest)
                    model_ys_test = sorted_y_test == curr_class;

                    % Implement the precision formula
                    nominator = cumsum(model_ys_test) .* model_ys_test;
                    denominator = (1:length(model_ys_test))';
                    average_precisions_all(i) = sum(nominator ./ denominator) / sum(model_ys_test);

                    results_folder = sprintf('results/%s_%s_%d_%d', colorspace, sampling_type, vocab_size, split_ratio);
                    if ~exist(results_folder, 'dir')
                        mkdir(results_folder)
                    end

                    figure(i)
                    suptitle(sprintf('Class %s', class_names{curr_class}));
                    for j=1:number_of_top_and_bottom_images
                        subplot(2, number_of_top_and_bottom_images, j)
                        imshow(squeeze(sorted_x_test(j,:,:,:)));
                        title(sprintf("Top #%d", j));

                        subplot(2, number_of_top_and_bottom_images, j+number_of_top_and_bottom_images)
                        imshow(squeeze(sorted_x_test(end-j-1,:,:,:)));
                        title(sprintf("Bottom #%d", j));
                    end
                    print(sprintf('%s/%s',results_folder, class_names{curr_class}),'-dpng');
                    close;    
                end

                mAP = mean(average_precisions_all);
                
                map_file = fopen(sprintf("%s/mAP.txt", results_folder), 'w');
                fprintf(map_file, "%f", mAP);
                fclose(map_file);
                
                ap_file = fopen(sprintf("%s/APs.txt", results_folder), 'w');
                for i = 1:unique_classes_size
                    curr_class = unique_classes(i);
                    fprintf(ap_file, "%s - %f \n", class_names{curr_class}, average_precisions_all(i));
                end
                fclose(ap_file);

                % populate the values for plotting the bars at the end
                vocab_size_bar_chart_values{vc} = [vocab_size_bar_chart_values{vc}, mAP];
                sampling_type_bar_chart_values{s} = [sampling_type_bar_chart_values{s}, mAP];
                colorspaces_bar_chart_values{c} = [colorspaces_bar_chart_values{c}, mAP];
                split_ratio_bar_chart_values{sr} = [split_ratio_bar_chart_values{sr}, mAP];
                
                vocab_size_bar_legend{vc} = [vocab_size_bar_legend{vc}, sprintf("%s-%s-%d",sampling_type, colorspace, split_ratio)];
                sampling_type_bar_legend{s} = [sampling_type_bar_legend{s}, sprintf("%s-%d-%d", colorspace, vocab_size, split_ratio)];
                colorspaces_bar_legend{c} = [colorspaces_bar_legend{c}, sprintf("%s-%d-%d", sampling_type, vocab_size, split_ratio)];
                split_ratio_bar_legend{sr} = [split_ratio_bar_legend{sr}, sprintf("%s-%s-%d",sampling_type, colorspace, vocab_size)];
            end
        end
    end
end

toc

bar_headers = categorical(sampling_types);
bar(bar_headers, cell2mat(sampling_type_bar_chart_values));
legend(sampling_type_bar_legend{1}, "Location", "eastoutside");
set(gcf, 'Position', [10 10 1200 600]);
saveas(gcf,"results/sampling_type_bar.png");
close; 

bar_headers = categorical(colorspaces);
bar(bar_headers, cell2mat(colorspaces_bar_chart_values));
legend(colorspaces_bar_legend{1}, "Location", "eastoutside");
set(gcf, 'Position', [10 10 1200 600]);
saveas(gcf,"results/colorspaces_bar.png");
close; 

bar_headers = categorical(vocab_sizes);
bar(bar_headers, cell2mat(vocab_size_bar_chart_values));
legend(vocab_size_bar_legend{1}, "Location", "eastoutside");
set(gcf, 'Position', [10 10 1200 600]);
saveas(gcf,"results/vocab_size_bar.png");
close; 

bar_headers = categorical(split_ratios);
bar(bar_headers, cell2mat(split_ratio_bar_chart_values));
legend(split_ratio_bar_legend{1}, "Location", "eastoutside");
set(gcf, 'Position', [10 10 1200 600]);
saveas(gcf,"results/split_ratio_bar.png");
close;

