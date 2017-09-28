%{
load('data_batch_1.mat')
train_x = double(data) / 255;
train_y = zeros(length(labels),2);
for i = 1:length(labels)
    for j = 1:10
        if labels(i) > 1 && labels(i) < 8
            train_y(i,2) = 1;
        else
            train_y(i,1) = 1;
        end
    end
end
load('data_batch_2.mat')
test_x = double(data) / 255;
test_y = zeros(length(labels),2);
for i = 1:length(labels)
    for j = 1:10
        if labels(i) > 1 && labels(i) < 8
            test_y(i,2) = 1;
        else
            test_y(i,1) = 1;
        end
    end
end
%}
%{
tic
%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
dbn.sizes = [100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights
t=toc
%}
tic
%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   10;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);


%%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);
nn.activation_function = 'sigm';

%%train nn
opts.numepochs =  40;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad, confmat] = nntest(nn, test_x, test_y);
er
confmat
precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
prec=precision(confmat)
rec=recall(confmat)
t2=toc
%bad
%assert(er < 0.10, 'Too big error');
