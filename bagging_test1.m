%{
load('data_batch_1.mat')
data = double(data) / 255;
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


load('test_batch.mat')
data_test = double(data) / 255;
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


n=10000;
numsamples=5000;
numbags=3;
%numsamples=n/numbags;

for j=1:numbags
    rs = randsample(1:n,numsamples);
    j
   % z(j,:)=rs';
    for i= 1:numsamples
        in=rs(:,i);
        tdata(i,:)=data(in,:);
        ttarget_train(i,:)=target_train_1(in,:);
    end
    
    %rng('shuffle')
    hiddenSize1 = 100;
    autoenc1 = trainAutoencoder(tdata.',hiddenSize1, ...
    'MaxEpochs',10, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

    feat1 = encode(autoenc1,tdata.');  % data -> input data (10,000 images)
    hiddenSize2 = 50;
    autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
        'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

    feat2 = encode(autoenc2,feat1);
    softnet = trainSoftmaxLayer(feat2,ttarget_train.','MaxEpochs',100);
    deepnet = stack(autoenc1,autoenc2,softnet);
%end
    
    y = deepnet(data_test.'); % data_test -> test input
    %plotconfusion(target_test.',y);  % target_test -> test data actual o/p
    
    deepnet = train(deepnet,tdata.',ttarget_train.');
    y = deepnet(data_test.');
    [c(j),~,~,~]=confusion(target_test.',y);
    %c=confusion(target_test.',y);
    %plotconfusion(target_test.',y);
end  
%}


