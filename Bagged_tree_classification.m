
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

load('data_batch_2.mat')
data_test = double(datat) / 255;
%target_test = zeros(length(labels),2);
for i = 1:length(labels)
    for j = 1:10
        if labels(i) > 1 && labels(i) < 8
            target_test(i,1) = 1;
        else
            target_test(i,1) = 2;
        end
    end
end
%}
%{
cv = cvpartition(y, 'holdout', .5);
Xtrain = X(cv.training,:);
Ytrain = y(cv.training,1);

Xtest = X(cv.test,:);
Ytest = y(cv.test,1);
%}
indices = crossvalind('Kfold','Var1',10);
cp = classperf(species);
for i = 1:10
    test = (indices == i); train = ~test;
    class = classify(meas(test,:),meas(train,:),species(train,:));
    classperf(cp,class,test)
end

mdl_ctree = ClassificationTree.fit(datat,target_train(:,1));
%ypred = predict(mdl_ctree,data_test);
ypred = predict(mdl_ctree,datat);
Confmat_ctree = confusionmat(target_train(:,1),ypred);
%{
mdl = fitensemble(datat,target_train(:,1),'bag',20,'tree','type','Classification');
ypred = predict(mdl,data_test);
Confmat_bag = confusionmat(target_test(:,1),ypred);
%}