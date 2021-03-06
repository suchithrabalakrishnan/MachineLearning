function [confmat]=autoenco(data,target_train,data_test,target_test)
    rng('default')
    hiddenSize1 = 100;
    autoenc1 = trainAutoencoder(data.',hiddenSize1, ...
    'MaxEpochs',5, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

    feat1 = encode(autoenc1,data.');  % data -> input data (10,000 images)
    hiddenSize2 = 50;
    autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',5, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
    feat2 = encode(autoenc2,feat1);
    softnet = trainSoftmaxLayer(feat2,target_train.','MaxEpochs',5);
    deepnet = stack(autoenc1,autoenc2,softnet);

    y = deepnet(data_test.'); % data_test -> test input
%plotconfusion(target_test.',y);  % target_test -> test data actual o/p

    deepnet = train(deepnet,data.',target_train.');
    y = deepnet(data_test.');
    confmat=confusionmat(target_test,y);
end
%plotconfusion(target_test.',y);
%}