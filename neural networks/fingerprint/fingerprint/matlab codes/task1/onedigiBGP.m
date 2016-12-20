% Training Digital sensor BGP features with Digital Test dataset
%Training data
trainBGP=[Train_All_Data_DigiBGP(:,1:916 )  Train_All_Data_DigiBGP(:,1016:1904 )]*10^8;
%Labelling Training data
trainlabelBGP=[Train_All_Label_DigiBGP(:,1:916) Train_All_Label_DigiBGP(:,1016:1904) ];
%Seperating Validation data from training data
validationBGP=[Train_All_Data_DigiBGP(:,916:1016) Train_All_Data_DigiBGP(:,1904:2004) ]10^8;
%Creating validation label
validationlabel=[Train_All_Label_DigiBGP(:,916:1016) Train_All_Label_DigiBGP(:,1904:2004)];
%Test data 
testBGP=[Test_All_Data_DigiBGP]10^8;
%Labelling test data
testlabelBGP=[Test_All_Label_DigiBGP];

%Auto encoder 1 perametres
hiddenSize1 =300;
autoenc1 = trainAutoencoder(trainBGP,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,trainBGP);
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
softnet = trainSoftmaxLayer(feat2,trainlabelBGP,'MaxEpochs',200);
deepnet = stack(autoenc1,autoenc2,softnet);
%Train deepnet
deepnet = train(deepnet,trainBGP,trainlabelBGP);
%Passing test and validation data to deepnet
ytrain = deepnet(trainBGP);
yvalidate = deepnet(validationBGP);
ytest = deepnet(testBGP);
% Plotting ezroc curves
ezroc3new(ytrain,trainlabelBGP,2,'',1);
ezroc3new(yvalidate,validationlabel,2,'',1);
ezroc3new(ytest,testlabelBGP,2,'',1);
