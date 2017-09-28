function [betaT,net]=boost(in,B,validation,tB,outfile,bmax,S1)
%(train_data,train_target,test_data,test_target,log_filename,numberoftimestoboost,numberofhiddenlayers)
% WBL 22 August 2002 Boost Matlab MLP Neural Network bmax times
% 1 input neuron per attribute, size(in)//1 hidden layer of S1 inputs//2 output neurons 
%Implement Pseudo-Loss ADAboost.M2 Table 1 p1873
%"R" p1874, Boosting Neural Networks, 

% expand B into targets. Each element becomes 2 element row,element corresponding to B(i) 
%is one all others are zero.% (already done for validation)
    ntests = size(B,2);
    S2 = 2; %number of output neurons
    targets = zeros(S2,ntests);
    for i=1:ntests
        targets(B(i)+1.0,i) = 1.0;
    end
    %weights used to decide which training data to use for boosted network
    %(data split into use for training and use for deciding when to stop training)
    weights = zeros(1,ntests);
    for i=1:ntests
        weights(i) = 1/ntests;
    end
    %outfile = fopen('boost25-sep-2002.log','w');
    fprintf(outfile,'#boost.m %s train %d, ver %d %s\n',versn,ntests,size(tB,2),datestr(now));
    %bmax=3;
    betaT=zeros(1,bmax);
    t=1;
    for b=1:bmax
        %Select training data for network using weights
        if(t>1)
            [index1,index2] = randselect(weights);
        else 
            %first time use all training data 
            index1=1:1:ceil(ntests/2); %test=in; y=B;
            index2=ceil(ntests/2)+1:1:ntests;
        end
        %gather statistics about data actually used for training
        [t1,t1p,t1n,u1,u1p,u1n] = numtests(targets,index1);
        n=hist(index1,[1:ntests]);
        fprintf(         '%d %s Range test(%d,%d,%d,%d,%d,%d) %d %d %d ',b,datestr(now),t1,t1p,t1n,u1,u1p,u1n,min(n),max(n),median(n));
        fprintf(outfile,'%4d %s Range test(%d,%d,%d,%d,%d,%d) %d %d %d ',b,datestr(now),t1,t1p,t1n,u1,u1p,u1n,min(n),max(n),median(n));
        [t2,t2p,t2n,u2,u2p,u2n] = numtests(targets,index2);
        n=hist(index2,[1:ntests]);
        fprintf(        'stop(%d,%d,%d,%d,%d,%d) %d %d %d ',t2,t2p,t2n,u2,u2p,u2n,min(n),max(n),median(n));
        fprintf(outfile,'stop(%d,%d,%d,%d,%d,%d) %d %d %d ',t2,t2p,t2n,u2,u2p,u2n,min(n),max(n),median(n));
    
        train     =      in(:,index1);
        trainans  = targets(:,index1);
        stopset.P =      in(:,index2);
        stopset.T = targets(:,index2);
    
        [net{t},epochs,rms]=trainnet(train,trainans,S1,stopset);
        net_tr = sim(net{t},in);
        %net_tr = simple(targets);
        [betaT(t),loss] = pseudoloss(weights,net_tr,targets);
        fprintf(        'epochs %3d RMS %f beta %f\n',epochs,rms,betaT(t));
        fprintf(outfile,'epochs %3d RMS %f beta %f ', epochs,rms,betaT(t));
    
        %NON STANDARD paper says stop if beta>1 but get such large stochastic effects on P450 data
        %would stop many time on first trail! Hence simply discard this trial - roll back and try again
    
        if(betaT(t)>=1) 
            fprintf(outfile,'Boosting trial %4d failed\n',b);
        else 
            [roc,net_tr] = ensemble(t,betaT,net,in,B,outfile);
            fprintf(outfile,'AUROC %f ',roc);
            [roc,net_vr] = ensemble(t,betaT,net,validation.P,tB',outfile);
            fprintf(outfile,'AUROC %f\n',roc);
    
            weights=update(weights,betaT(t),loss);
            t = t+1;
            fprintf(outfile,'#%4d %4d Range of weights min %f (%f) max %f (%f)\n',b,t,...
                min(weights), min(weights)/mean(weights),...
                max(weights), max(weights)/mean(weights));
        end
end
%save('boost','versn','betaT','net');
%fclose (outfile);