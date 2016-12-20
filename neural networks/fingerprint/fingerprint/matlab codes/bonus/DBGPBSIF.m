%Training fusion of BGP+BSIF features from Digital sensor and testing on the same sensor 

%Training concatination of features from BGP and BSIF leaving 200 values for live and spoof vector values to use for validation 
trainBGPBSIF=[Train_All_Data_DigiBGP(:,1:1000) Train_All_Data_DigiBGP(:,1101:1900) *10^8; Train_All_Data_DigiBSIF(:,1:1000 ) Train_All_Data_DigiBSIF(:,1101:1900)*10^8];
%Creating label for training data
trainlabelBGPBSIF=[Train_All_Label_DigiBGP(:,1:1000) Train_All_Label_DigiBGP(:,1101:1900); Train_All_Label_DigiBSIF(:,1:1000) Train_All_Label_DigiBSIF(:,1101:1900)];
%Validation data
validateBGPBSIF=[Train_All_Data_DigiBGP(:,1001:1100)  Train_All_Data_DigiBGP(:,1901:2004) *10^8; Train_All_Data_DigiBSIF(:,1001:1100) Train_All_Data_DigiBSIF(:,1901:2004)*10^8];
%Validation data label
validatelabelBGPBSIF=[Train_All_Label_DigiBGP(:,1001:1100)  Train_All_Label_DigiBGP(:,1901:2004); Train_All_Label_DigiBSIF(:,1001:1100) ,Train_All_Label_DigiBSIF(:,1901:2004)];

%Testing concatination of features from BGP and BSIF leaving 200 values for live and spoof vector values to use for validation
testBGPBSIF=[Test_All_Data_DigiBSIF*(10^8) ;Test_All_Data_DigiBGP]; 
%Labelling test data
testlabelBGPBSIF=[Test_All_Label_DigiBSIF ;Test_All_Label_DigiBGP];

%Auto encoder 1 perametres
hiddenSize1 =300;
autoenc1 = trainAutoencoder(trainBGPBSIF,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',0.04, ...
    'SparsityProportion',0.2, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,trainBGPBSIF);

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
softnet = trainSoftmaxLayer(feat2,trainlabelBGPBSIF,'MaxEpochs',200);
deepnet = stack(autoenc1,autoenc2,softnet);
%Training deep neural network
deepnet = train(deepnet,trainBGPBSIF,trainlabelBGPBSIF);
%Creating validation and test data to deep neural network
ytrain = deepnet(trainBGPBSIF);
yvalidate = deepnet(validateBGPBSIF);
ytest = deepnet(testBGPBSIF);
%Plotting ezroc values
ezroc3new(ytrain,trainlabelBGPBSIF,2,'',1);
ezroc3new(yvalidate,validatelabelBGPBSIF,2,'',1);
ezroc3new(ytest,testlabelBGPBSIF,2,'',1);