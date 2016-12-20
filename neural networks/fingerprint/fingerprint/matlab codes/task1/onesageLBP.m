% Training Sagem sensor LBP features with Sagem Test dataset

%Training data
trainLBP=[Train_All_Data_SageLBP(:,1:916 )  Train_All_Data_SageLBP(:,1016:1916 )];
%Labelling Training data
trainlabelLBP=[Train_All_Label_SageLBP(:,1:916) Train_All_Label_SageLBP(:,1016:1916) ];
%Seperating Validation data from training data
validationLBP=[Train_All_Data_SageLBP(:,916:1016)];
%Creating validation label
validationlabel=[Train_All_Label_SageLBP(:,916:1016)];
%Test data 
testLBP=[Test_All_Data_SageLBP Train_All_Data_SageLBP(:,1916:2016)];
%Labelling test data
testlabelLBP=[Test_All_Label_SageLBP Train_All_Label_SageBSIF(:,1916:2016)];

%Auto encoder 1 perametres
hiddenSize1 =300;
autoenc1 = trainAutoencoder(trainLBP,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,trainLBP);
%Auto encoder 1 perametres
hiddenSize2 = 100;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
plotWeights(autoenc2);
feat2 = encode(autoenc2,feat1);
% Training softamx layer and stacking all features created
softnet = trainSoftmaxLayer(feat2,trainlabelLBP,'MaxEpochs',200);
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