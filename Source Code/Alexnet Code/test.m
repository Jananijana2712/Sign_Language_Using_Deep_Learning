% % % % % % % % % % % % % % % % % % % % %
% % % % % %TEST THE IMAGES % % % % % % %
% % % % % % % % % % % % % % % % % % % % % %
% % 


[filename,pathname] = uigetfile('*.png*');

im = imread([pathname,filename]);

figure,imshow(im),title('INPUT IMAGE');

% % % % % % % TO RESIZE THE INPUT IMAGE % % %

I = imresize(im,[227, 227]);

% % % % % % % OUTPUT CLASSIFICATION% % % % % % % % %

output=classify(training,I);

figure, imshow(I); title(string(output));

msgbox(string(output));
% % % % % % % % % % % % % % % % % % % % % % % % % % % %