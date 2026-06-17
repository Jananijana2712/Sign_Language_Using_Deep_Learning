% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % SIGN LANGUAGE USING TRANSFER LEARNING % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
clc;
 clear all;

close all;

warning off;


% % % % % % % % % % % % % % % % % % % %
% % % % %TRAIN THE IMAGES % % % % % % %
% % % % % % % % % % % % % % % % % % % %

% % % % % SET TRAINING DATASET FOLDER % % % % % % %
matlabpath = 'C:\Users\JANANI\Downloads\sign language';

data = fullfile(matlabpath,'TRAINING IMAGES');

train = imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames');

count  = train.countEachLabel;


% % % % % % % % SET VALIDATION DATASET FOLDER% % % % % % % %

validationPath = fullfile(matlabpath,'TESTING IMAGES');

imdsValidation = imageDatastore(validationPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% % % % % LOAD THE NETWORK % % % % % % %

net = alexnet;

layers = [
    imageInputLayer([227 227 3]);
    
    net(2:end-3);
    
    fullyConnectedLayer(26);
    
    softmaxLayer;
    
    classificationLayer();
    
    ];


% % % % % %TRAINING  % % % % %

options=trainingOptions('sgdm','MaxEpochs',5,'InitialLearnRate',0.0001,'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30,...
    'Verbose',false, ...
    'Plots','training-progress');


% % % % TRAIN NETWORK USING TRAINING DATA % % % % %

training = trainNetwork(train,layers,options);

% % % % CLASSIFY VALIDATION IMAGES AND COMPUTE ACCURACY % % % % %

YPred = classify(training,imdsValidation);

YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);

analyzeNetwork(training);

save training.mat training;
load training.mat;

