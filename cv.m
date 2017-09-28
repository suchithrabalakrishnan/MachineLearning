%{
load('data_batch_1.mat')
datat = double(data) / 255;
%target_train = zeros(length(labels),2);
for i = 1:length(labels)
    for j = 1:10
        if labels(i) > 1 && labels(i) < 8
            target_train(i,1) = 1;
        else
            target_train(i,1) = 2;
        end
    end
end
%}
indices = crossvalind('Kfold',target_test(:,1),10);
cp = classperf(target_test(:,1));
%{
for i = 1:10
    test = (indices == i); train = ~test;
    class = classify(meas(test,:),meas(train,:),species(train,:));
    classperf(cp,class,test)
end
%}
