train_BGP=[Train_All_Data_DigiBGP(:,1:916 ) Train_All_Data_DigiBGP(:,1016:1916) Train_All_Data_SageBGP(:,1:916 ) Train_All_Data_SageBGP(:,1016:1916 )]*10^8;
trainlabel_BGP=[Train_All_Label_DigiBGP(:,1:916) Train_All_Label_DigiBGP(:,1016:1916) Train_All_Label_SageBGP(:,1:916) Train_All_Label_SageBGP(:,1016:1916)];
testBGP=[Test_All_Data_DigiBGP Test_All_Data_SageBGP]*10^8;
testlabelBGP=[Test_All_Label_DigiBGP Test_All_Label_SageBGP];

hiddenSize1 =300;
autoenc1 = trainAutoencoder(train_BGP,hiddenSize1, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',1, ...
    'SparsityProportion',0.8, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,train_BGP);
hiddenSize2 = 100;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.008, ...
    'SparsityRegularization',1, ...
    'SparsityProportion',0.8, ...
    'ScaleData', false);
plotWeights(autoenc2);
feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2,trainlabel_BGP,'MaxEpochs',100);
deepnet = stack(autoenc1,autoenc2,softnet);
deepnet = train(deepnet,train_BGP,trainlabel_BGP);

ytrain = deepnet(train_BGP);
ytest = deepnet(testBGP);
ezroc3new(ytest,testlabelBGP,2,'',1);