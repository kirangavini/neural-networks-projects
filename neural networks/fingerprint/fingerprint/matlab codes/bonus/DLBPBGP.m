%Training fusion of BGP+LBP features from Digital sensor and testing on the same sensor


%Training concatination of features from BGP and LBP leaving 200 values for live and spoof vector values to use for validation 
trainLBPBGP=[Train_All_Data_DigiLBP(:,1:1000) Train_All_Data_DigiLBP(:,1101:1900) ; Train_All_Data_DigiBGP(:,1:1000 ) Train_All_Data_DigiBGP(:,1101:1900)*10^8];
%Creating label for training data
trainlabelLBPBGP=[Train_All_Label_DigiLBP(:,1:1000) Train_All_Label_DigiLBP(:,1101:1900); Train_All_Label_DigiBGP(:,1:1000) Train_All_Label_DigiBGP(:,1101:1900)];
%Validation data
validateLBPBGP=[Train_All_Data_DigiLBP(:,1001:1100)  Train_All_Data_DigiLBP(:,1901:2004); Train_All_Data_DigiBGP(:,1001:1100) Train_All_Data_DigiBGP(:,1901:2004)*10^8];
%Validation data label
validatelabelLBPBGP=[Train_All_Label_DigiLBP(:,1001:1100)  Train_All_Label_DigiLBP(:,1901:2004); Train_All_Label_DigiBGP(:,1001:1100) ,Train_All_Label_DigiBGP(:,1901:2004)];

%Testing concatination of features from BGP and LBP leaving 200 values for live and spoof vector values to use for validation 
testLBPBGP=[Test_All_Data_DigiBGP*(10^8) ;Test_All_Data_DigiLBP]; 
% Assigning test label
testlabelLBPBGP=[Test_All_Label_DigiBGP ;Test_All_Label_DigiLBP];

%Auto encoder 1 perametres
hiddenSize1 =300;
autoenc1 = trainAutoencoder(trainLBPBGP,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',0.04, ...
    'SparsityProportion',0.2, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,trainLBPBGP);

%Auto encoder 2 perametres
hiddenSize2 = 100;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',0.04, ...
    'SparsityProportion',0.2, ...
    'ScaleData', false);
plotWeights(autoenc2);
feat2 = encode(autoenc2,feat1);

%Training softmax classifier and stacking them
softnet = trainSoftmaxLayer(feat2,trainlabelLBPBGP,'MaxEpochs',200);
deepnet = stack(autoenc1,autoenc2,softnet);
%Training deep neural network
deepnet = train(deepnet,trainLBPBGP,trainlabelLBPBGP);

%Creating validation and test data to deep neural network
ytrain = deepnet(trainLBPBGP);
yvalidate = deepnet(validateLBPBGP);
ytest = deepnet(testLBPBGP);

%Plotting ezroc values
ezroc3new(ytrain,trainlabelLBPBGP,2,'',1);
ezroc3new(yvalidate,validatelabelLBPBGP,2,'',1);
ezroc3new(ytest,testlabelLBPBGP,2,'',1);