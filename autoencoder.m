target_train_1 = zeros(length(labels),2);
for i = 1:length(labels)
    for j = 1:10
        if labels(i) > 1 && labels(i) < 8
            target_train_1(i,2) = 1;
        else
            target_train_1(i,1) = 1;
        end
    end
end

target_test = zeros(length(labels_test),2);
for i = 1:length(labels_test)
    for j = 1:10
        if labels_test(i) > 1 && labels_test(i) < 8
            target_test(i,2) = 1;
        else
            target_test(i,1) = 1;
        end
    end
end

rng('default')
hiddenSize1 = 400;
autoenc1 = trainAutoencoder(data.',hiddenSize1, ...
    'MaxEpochs',50, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
%view(autoenc1)
%figure()
%plotWeights(autoenc1);
feat1 = encode(autoenc1,data.');  % data -> input data (10,000 images)
hiddenSize2 = 200;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',10, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
%view(autoenc2);
%figure()
%plotWeights(autoenc2);

feat2 = encode(autoenc2,feat1);
softnet = trainSoftmaxLayer(feat2,target_train_1.','MaxEpochs',100);
%view(softnet);
deepnet = stack(autoenc1,autoenc2,softnet);
%view(deepnet);


y = deepnet(data_test.'); % data_test -> test input
plotconfusion(target_test.',y);  % target_test -> test data actual o/p

deepnet = train(deepnet,data.',target_train_1.');
y = deepnet(data_test.');
plotconfusion(target_test.',y);
