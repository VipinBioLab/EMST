clc,clear,
close all;
Input_Image=imread('D:\NITPY PhD\Contrast_EnH_Skin_Images\Test_Data_New\2_Img.jpg');
Ground_Truth=imread('D:\NITPY PhD\Contrast_EnH_Skin_Images\Test_Data_New\2_Msk.png');
%Ground_Truth=im2bw(imread('D:\NITPY PhD\Contrast_EnH_Skin_Images\Test_Data_New\14_Msk.png'),.5);
%%%Convert the input image from RGB colour space to HSV colour space%%%
HSV_Image = rgb2hsv(Input_Image);
%%%Seperate Hue, Saturation and Value Components%%%
Hue_Component=HSV_Image(:,:,1);
Saturation_Component=HSV_Image(:,:,2);
Value_Component=HSV_Image(:,:,3);
Value_Component_Double=double(Value_Component);
Shape_Parameter=20;
Crossover_Point_Array=(unique(Value_Component_Double(:)))';
for Crossover_Point_Index=2:length(Crossover_Point_Array)-1
    Crossover_Point=Crossover_Point_Array(Crossover_Point_Index);
%%%Sigmoid Transform%%%
Enhanced_Value=1./(1+(exp(Shape_Parameter*(Crossover_Point-Value_Component_Double))));
%%%Create a 2D array for enhanced image%%%
Enhanced_HSV_Image=HSV_Image;
%%%Insert the enhanced value component%%%
Enhanced_HSV_Image(:,:,3)=uint8(Enhanced_Value);
%%%HSV to RGB colour space conversion%%%
Enhanced_RGB_Image = hsv2rgb(Enhanced_HSV_Image);
Contrast_Enhanced_Image=uint8(255*Enhanced_RGB_Image);
Enhanced_Image_Gray=rgb2gray(Contrast_Enhanced_Image);
%%%Otsu's Thresholding%%%%
Threshold_Value = graythresh(Enhanced_Image_Gray);
% %%%The function gives normalized Threshold%%%
Threshold_Value=Threshold_Value*255;
Segmented_Output=Enhanced_Image_Gray<Threshold_Value;
DSI_Otsu(Crossover_Point_Index-1) = dice(Segmented_Output,Ground_Truth);
end
Array_Cross_Over_Points=Crossover_Point_Array(2:end-1);
figure,plot(Array_Cross_Over_Points,DSI_Otsu);
xlabel('Crossover Point');
ylabel('Dice Similarity Coefficient');
%%%Find Optimum Value of Compensation Parameter from Dice of Otsu%%%
Opt_Crossover_Index=find(DSI_Otsu==max(DSI_Otsu))
Opt_Crossover_Point=Array_Cross_Over_Points(Opt_Crossover_Index)
%%%Please write Value_Component_Double or Value_Component_Uint8 to desired location using imwrite%%%
Value_Component_Double=Value_Component;
Value_Component_Uint8=uint8(Value_Component*255);
%imwrite(Value_Component_Double,'D:\NITPY PhD\Contrast_EnH_Skin_Images\Source_Codes\Train_images\601.jpg'); 
%xlswrite('Optimum value_train.xls',Opt_Crossover_Point,'Optimum value','A601');
