Data = imageSet('orl_faces','recursive');
data=cell(1,240);  % Create a cell for 40x6 Training images
a = 1;
for j=1:40
    for i=1:6;
        X= read(Data(j),i);     % Reading 240 images
        X=double(X)/256;
        data{a} = X;
        a = a + 1;
    end;
end;
rng('default');
% Creating auto encoder
hiddenSize1 = 1000;
autoenc1 = trainAutoencoder(data,hiddenSize1, ...         %Train encoder
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.006, ...
    'SparsityRegularization',1, ...
    'SparsityProportion',0.25, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,data);
intra=zeros(1,600);
k=1;
% Creating a mse for inter and intra classes
for guide1=0:6:234;
    for i= 1:1:6      % Images are compared with other images in the same group
        for j=2:1:6
            if i<j
                    intra(1,k)=(mse(feat1(:,i+guide1),feat1(:,j+guide1)));
                     k=k+1;
            end;
        end;
    end;
end;
l=1;
inter=zeros(1,28080);
for guide1=0:6:234        % Images are compared with other images in different group
    for guide2=0:6:234;
        if(guide1<guide2)
            for i=1:1:6
                for  j=1:1:6;
                    inter(1,l)=(mse(feat1(:,i+guide1),feat1(:,j+guide2)));
                          l=l+1;
                end;
            end;
        end;
    end;
end;

ratio=sum(inter(:))/sum(intra(:)); % Ratio
H=[-1*inter -1*intra];    % Attaching both classes and inversing the values
T=[zeros(1,28080),ones(1,600)];
value=ezroc3(H,T,2,' ',1);
