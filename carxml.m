
negativeimages = imageDatastore('negative images','FileExtensions',{'.jpg','.jfif','.png'},'IncludeSubfolders',true,'LabelSource','foldernames');
%%
carimages = imageDatastore('car images','FileExtensions',{'.jpg','.jfif','.png'},'IncludeSubfolders',true,'LabelSource','foldernames');

qi = readimage(carimages,15);
qi = imresize(qi,[400 400]);
qi  =rgb2gray(qi);
[hog,vis] = extractHOGFeatures(qi,'CellSize',[9 9]);
imshow(qi); hold on; plot(vis);
numImages = numel(carimages.Files);
trainingFeatures=  zeros(numImages,length(hog),'single');
%%
for i = 1:numImages
    im = readimage(carimages,i);
    im = imresize(im,[400 400]);
    trainingFeatures(i,:) = extractHOGFeatures(im,'CellSize',[9 9]);
end
%%
trainCascadeObjectDetector('CARXMLFILEHOGFEATS.xml',CarData,negativeimages,'FeatureType','Haar','TruePositiveRate',0.92,'NumCascadeStages',30,'FalseAlarmRate',0.7);
%%
clc
trainCascadeObjectDetector('CARXMLFILEHOGFEATS.xml',CarData,negativeimages,'TruePositiveRate',0.92,'NumCascadeStages',30,'FalseAlarmRate',0.7);
%%
image
detector = vision.CascadeObjectDetector('CARXMLFILEHOGFEATS.xml','ScaleFactor',1.00505,'MergeThreshold',7);
%[carimage,info] = readimage(imds,21); imshow(carimage);
carimage = imread('IMG_20200303_103132_9.jpg');

bbox = step(detector,carimage);
imshow(insertShape(carimage,'rectangle',bbox))

%%

release(videoreader)
videoreader = vision.VideoFileReader('cad1.mp4');
%%
videoplayer = vision.VideoPlayer();
release(detector)
while ~isDone(videoreader)
   frame = step(videoreader);
   bbox1 = step(detector,frame);
   j = insertShape(frame,'rectangle',bbox1); 
    step(videoplayer,j)
end

 