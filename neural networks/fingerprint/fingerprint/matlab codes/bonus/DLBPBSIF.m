%Training fusion of LBP+BSIF features from Digital sensor and testing on the same sensor

%Training concatination of features from LBP and BSIF leaving 200 values for live and spoof vector values to use for validation 
trainLBPBSIF=[Train_All_Data_DigiLBP(:,1:1000) Train_All_Data_DigiLBP(:,1101:1900) ; Train_All_Data_DigiBSIF(:,1:1000 ) Train_All_Data_DigiBSIF(:,1101:1900)*10^8];
%Creating label for training data
trainlabelLBPBSIF=[Train_All_Label_DigiLBP(:,1:1000) Train_All_Label_DigiLBP(:,1101:1900); Train_All_Label_DigiBSIF(:,1:1000) Train_All_Label_DigiBSIF(:,1101:1900)];
%Validation data
validateLBPBSIF=[Train_All_Data_DigiLBP(:,1001:1100)  Train_All_Data_DigiLBP(:,1901:2004); Train_All_Data_DigiBSIF(:,1001:1100) Train_All_Data_DigiBSIF(:,1901:2004)*10^8];
%Validation data label
validatelabelLBPBSIF=[Train_All_Label_DigiLBP(:,1001:1100)  Train_All_Label_DigiLBP(:,1901:2004); Train_All_Label_DigiBSIF(:,1001:1100) ,Train_All_Label_DigiBSIF(:,1901:2004)];

%Testing concatination of features from LBP and BSIF leaving 200 values for live and spoof vector values to use for validation 
testLBPBSIF=[Test_All_Data_DigiBSIF*(10^8) ;Test_All_Data_DigiLBP]; 
%Labelling Test features
testlabelLBPBSIF=[Test_All_Label_DigiBSIF ;Test_All_Label_DigiLBP];

%Auto encoder 1 perametres
hiddenSize1 =300;
autoenc1 = trainAutoencoder(trainLBPBSIF,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',0.04, ...
    'SparsityProportion',0.2, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,trainLBPBSIF);


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
softnet = trainSoftmaxLayer(feat2,trainlabelLBPBSIF,'MaxEpochs',200);
deepnet = stack(autoenc1,autoenc2,softnet);
%Training deep neural network
deepnet = train(deepnet,trainLBPBSIF,trainlabelLBPBSIF);

%Creating validation and test data to deep neural network
ytrain = deepnet(trainLBPBSIF);
yvalidate = deepnet(validateLBPBSIF);
ytest = deepnet(testLBPBSIF);

%Plotting ezroc values
ezroc3new(ytrain,trainlabelLBPBSIF,2,'',1);
ezroc3new(yvalidate,validatelabelLBPBSIF,2,'',1);
ezroc3new(ytest,testlabelLBPBSIF,2,'',1);