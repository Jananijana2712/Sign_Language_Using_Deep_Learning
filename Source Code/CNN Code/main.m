%% 
%%%%%%%%%%% SIGN LANGUAGE %%%%%%%%%%%%%%

clc;

clear;

close all;

warning off

% % TRAIN THE DATASET IMAGES %%%%%
  
matlabroot='C:\Users\JANANI\Downloads\SIGN LANGUGAE';

data1 = fullfile(matlabroot,'TRAINING IMAGES');
Data=imageDatastore(data1,'IncludeSubfolders',true,'LabelSource','foldernames');

validationPath = fullfile(matlabroot,'TESTING IMAGES');
imdsValidation = imageDatastore(validationPath, ...    
'IncludeSubfolders',true,'LabelSource','foldernames');

 % % CREATE CONVOLUTIONAL NEURAL NETWORK LAYERS 


 layers=[imageInputLayer([227 227 3])  
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)    
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(26)
    softmaxLayer
    classificationLayer];

options=trainingOptions('sgdm','MaxEpochs',10,'InitialLearnRate',0.00001,'Shuffle','every-epoch', ...    
    'ValidationData',imdsValidation, ...        
    'ValidationFrequency',30,...        
    'Verbose',false, ...        
    'Plots','training-progress');

 convnet=trainNetwork(Data,layers,options);

 save convnet.mat convnet



% CLASSIFY VALIDATION IMAGES AND COMPUTE ACCURACY % % % % %

YPred = classify(convnet,imdsValidation);

YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);



% %  %%%%%%%%%%READ THE IMAGE FROM THE DATASET %%%%%%%%%%%
% % % % % % % % 
%  load convnet.mat
% [filename,pathname]=uigetfile('*.*'); 
% 
% im1=imread([pathname,filename]);
% 
% figure,imshow(im1),title('INPUT IMAGE');
% 
% 
% %%%%% RESIZE THE IMAGE %%%%%%%%%%
% 
%  
% im=imresize(im1,[227 227]);
% 
% figure,imshow(im),title('Resized image');
% 
% %%%%%%%%%%% CONVERT THE DATA TYPE INTO UNSIGNED INTEGER %%%%%%%%%%%
%  re=im2uint8(im);
% 
% 
% %% TO  CLASSIFY THE OUTPUT %%%%%%% 
% 
% output=classify(convnet,re);
% tf1=[];
% 
% for ii=1:26
%     st=int2str(ii)
%     tf=ismember(output,st);
%     tf1=[tf1 tf];
% end
% output1=find(tf1==1);
% 
% if output1==1
%      
%     msgbox('a')
%     
% elseif output1==2
%      
%     msgbox('b')
% elseif output1==3
%      
%     msgbox('c')
% elseif output1==4
%      
%     msgbox('d')
% elseif output1==5
%      
%     msgbox('e')
% elseif output1==6
%      
%     msgbox('f')
% elseif output1==7
%      
%     msgbox('g')
% elseif output1==8
%      
%     msgbox('h')
% elseif output1==9
%      
%     msgbox('i')
% elseif output1==10
%      
%     msgbox('j')
% elseif output1==11
%      
%     msgbox('k')
% elseif output1==12
%      
%     msgbox('l')
% elseif output1==13
%      
%     msgbox('m')
% elseif output1==14
%      
%     msgbox('n')
% elseif output1==15
%      
%     msgbox('o')
% elseif output1==16
%      
%     msgbox('p')
% elseif output1==17
%      
%     msgbox('q')
% elseif output1==18
%      
%     msgbox('r')
% elseif output1==19
%      
%     msgbox('s')
% elseif output1==20
%      
%     msgbox('t')
% elseif output1==21
%      
%     msgbox('u')
%     elseif output1==22
%      
%     msgbox('v')
% elseif output1==23
%      
%     msgbox('w')
% elseif output1==24
%      
%     msgbox('x')
% elseif output1==25
%      
%     msgbox('y')
% elseif output1==26
%      
%     msgbox('z')
% end
