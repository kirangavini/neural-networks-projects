% Training Digital sensor LBP features with Digital Test dataset
%Training data
trainLBP=[Train_All_Data_DigiLBP(:,1:916 )  Train_All_Data_DigiLBP(:,1016:1904 )];
%Labelling Training data
trainlabelLBP=[Train_All_Label_DigiLBP(:,1:916) Train_All_Label_DigiLBP(:,1016:1904) ];
%Seperating Validation data from training data
validationLBP=[Train_All_Data_DigiLBP(:,916:1016) Train_All_Data_DigiLBP(:,1904:2004) ];
%Creating validation label
validationlabel=[Train_All_Label_DigiLBP(:,916:1016) Train_All_Label_DigiLBP(:,1904:2004)];
%Test data 
testLBP=[Test_All_Data_DigiLBP];
%Labelling test data
testlabelLBP=[Test_All_Label_DigiLBP];

%Auto encoder 1 perametres
hiddenSize1 =110;
autoenc1 = trainAutoencoder(trainLBP,hiddenSize1, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,trainLBP);
%Auto encoder 1 perametres
hiddenSize2 = 100;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
plotWeights(autoenc2);
feat2 = encode(autoenc2,feat1);
% Training softamx layer and stacking all features created
softnet = trainSoftmaxLayer(feat2,trainlabelLBP,'MaxEpochs',100);
deepnet = stack(autoenc1,autoenc2,softnet);
%Train deepnet
deepnet = train(deepnet,trainLBP,trainlabelLBP);
%Passing test and validation data to deepnet
ytrain = deepnet(trainLBP);
yvalidate = deepnet(validationLBP);
ytest = deepnet(testLBP);
% Plotting ezroc curves
ezroc3new(ytrain,trainlabelLBP,2,'',1);
ezroc3new(yvalidate,validationlabel,2,'',1);
ezroc3new(ytest,testlabelLBP,2,'',1);
