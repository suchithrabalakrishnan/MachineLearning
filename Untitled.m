%{
load('data_batch_1.mat')
datat = double(data) / 255;
target_train = zeros(length(labels),2);
for i = 1:length(labels)
    for j = 1:10
        if labels(i) > 1 && labels(i) < 8
            target_train(i,2) = 1;
        else
            target_train(i,1) = 1;
        end
    end
end
%}
%{
load('data_batch_2.mat')
data_test = double(datat) / 255;
target_test = zeros(length(labels),2);
for i = 1:length(labels)
    for j = 1:10
        if labels(i) > 1 && labels(i) < 8
            target_test(i,2) = 1;
        else
            target_test(i,1) = 1;
        end
    end
end
%}

%classi=svmtrain(datat, target_train(:,1));
classifier = fitcecoc(datat, target_train(:,1));

%predictedLabels = svmclassify(classi, data_test);
%predictedLabels = predict(classi, data_test(1:100,:));
%confMat = confusionmat(target_test, predictedLabels);
%conMat
