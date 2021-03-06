% Training Digital sensor BSIF features with Digital Test dataset
%Training data
trainBSIF=[Train_All_Data_DigiBSIF(:,1:916 )  Train_All_Data_DigiBSIF(:,1016:1904 )]*10^8;
%Labelling Training data
trainlabelBSIF=[Train_All_Label_DigiBSIF(:,1:916) Train_All_Label_DigiBSIF(:,1016:1904) ];
%Seperating Validation data from training data
validationBSIF=[Train_All_Data_DigiBSIF(:,916:1016) Train_All_Data_DigiBSIF(:,1904:2004) ]10^8;
%Creating validation label
validationlabel=[Train_All_Label_DigiBSIF(:,916:1016) Train_All_Label_DigiBSIF(:,1904:2004)];
%Test data 
testBSIF=[Test_All_Data_DigiBSIF]10^8;
%Labelling test data
testlabelBSIF=[Test_All_Label_DigiBSIF];

%Auto encoder 1 perametres
hiddenSize1 =300;
autoenc1 = trainAutoencoder(trainBSIF,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,trainBSIF);
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
softnet = trainSoftmaxLayer(feat2,trainlabelBSIF,'MaxEpochs',200);
deepnet = stack(autoenc1,autoenc2,softnet);
%Train deepnet
deepnet = train(deepnet,trainBSIF,trainlabelBSIF);
%Passing test and validation data to deepnet
ytrain = deepnet(trainBSIF);
yvalidate = deepnet(validationBSIF);
ytest = deepnet(testBSIF);
% Plotting ezroc curves
ezroc3new(ytrain,trainlabelBSIF,2,'',1);
ezroc3new(yvalidate,validationlabel,2,'',1);
ezroc3new(ytest,testlabelBSIF,2,'',1);
