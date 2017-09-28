
data = double(data) / 255;
[coeff,~,~,~,explained,~] = pca(data);
ex=0; npca=0;
for i=1:3072
    ex=ex+explained(i);
    npca=npca+1;
    if ex>99
        break
    end
end
numberOfDimensions = 100;  
reducedDimension = coeff(:,1:numberOfDimensions);
reducedData = data * reducedDimension;

reddata=uint8(reducedData);

data_test = double(data_test) / 255;
coeff1 = pca(data_test);


numberOfDimensions = 100;  
reducedDimension = coeff1(:,1:numberOfDimensions);
reducedData_test = data_test * reducedDimension;
