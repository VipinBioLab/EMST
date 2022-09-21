%%%An EfficientNet-based modified sigmoid transform (EMST) for enhancing dermatological macro-images of melanoma and nevi skin lesions%%%
%%%Typical Usage%%%
%%%Contrast_Enhanced_Image=STEC(Input_Image,Shape_Parameter,Compensation_Parameter)%%%
%%%Inputs:%%%
%%%1. Input_Image: Input Colour Image (RGB), uint8 with range 0- 255%%%
%%%2. Shape_Parameter: User-defined parameteter that controls shape of the sigmoid transform, Default value of the Shape_Parameter is 20%%%
%%%3. Compensation_Parameter: Feed the cross-over point obtained from thr modified EfficientNet Regressor%%%
%%%Outputs%%%
%%%Contrast_Enhanced_Image: Contrast-enhanced Colour Image (RGB), uint8 with range 0- 255%%%
%%%Users please cite the paper: Vipin Venugopal, Justin Joseph, M. Vipin Das, Malaya Kumar Nath, "An EfficientNet-based modified sigmoid transform for enhancing dermatological macro-images of melanoma and nevi skin lesions", 
% Computer Methods and Programs in Biomedicine,Volume 222, 2022, 106935, https://doi.org/10.1016/j.cmpb.2022.106935.%%%%
%%%This code is shared for non-commercial ethical research%%%%%
%%%For any quiries please contact josephjusti@gmail.com or vipinscms@gmail.com%%%
clc,clear,
close all;
Input_Image=imread('D:\NITPY PhD\Contrast_EnH_Skin_Images\Test_Data_New\26_Img.jpg');
figure,imshow(Input_Image,'Border','tight');
Enhanced_Image=Sig_Tran_Manual_Cross_Over(Input_Image,20,0.20);
figure,imshow(Enhanced_Image,'Border','tight');
Enhanced_Image_Gray=rgb2gray(Enhanced_Image);
figure,imshow(Enhanced_Image_Gray,'Border','tight');
%%%Otsu's Thresholding%%%%
Threshold_Value = graythresh(Enhanced_Image_Gray);
% %%%The function gives normalized Threshold%%%
Threshold_Value=Threshold_Value*255
Segmented_Output=Enhanced_Image_Gray<Threshold_Value;
figure,imshow(Segmented_Output,'Border','tight');
Ground_Truth=imread('D:\NITPY PhD\Contrast_EnH_Skin_Images\Test_Data_New\26_Msk.png');
figure,imshow(Ground_Truth,'Border','tight');
DSI_Otsu = dice(Segmented_Output,Ground_Truth)
%%%Isodata Thresholding%%%
% Threshold_Value = isodata(Enhanced_Image_Gray);
% %%%The function gives normalized Threshold%%%
% Threshold_Value=Threshold_Value*255
% Segmented_Output=Enhanced_Image_Gray<Threshold_Value;
% figure,imshow(Segmented_Output,'Border','tight');
% DSI_Isodata = dice(Segmented_Output,Ground_Truth)
% 
% xlswrite('DSI_DL.xls',DSI_Otsu,'DSI_Otsu','A50');
% xlswrite('DSI_DL.xls',DSI_Isodata,'DSI_Isodata','A50');