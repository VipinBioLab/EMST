function Contrast_Enhanced_Image=Sig_Tran_Manual_Cross_Over(Input_Image,Shape_Parameter, Optimum_Crossover_Point)

%%%Convert the input image from RGB colour space to HSV colour space%%%

HSV_Image = rgb2hsv(Input_Image);

%%%Seperate Hue, Saturation and Value Components%%%

Hue_Component=HSV_Image(:,:,1);

Saturation_Component=HSV_Image(:,:,2);

Value_Component=HSV_Image(:,:,3);

Value_Component_Double=double(Value_Component);

%%%Sigmoid Transform%%%

Enhanced_Value=1./(1+(exp(Shape_Parameter*(Optimum_Crossover_Point-Value_Component_Double))));

%%%Normalization to the actual range does not have any impact%%%

%Enhanced_Value=((((Enhanced_Value-min(Enhanced_Value(:)))/(max(Enhanced_Value(:))-min(Enhanced_Value(:)))))*(max(Value_Component_Double(:))-min(Value_Component_Double(:))))+min(Value_Component_Double(:));

%%%Plot the Transformation Curve%%%

X_Axis=(unique(Value_Component_Double))';

Y_Axis=1./(1+(exp(Shape_Parameter*(Optimum_Crossover_Point-X_Axis))));

figure,plot(X_Axis,Y_Axis);

xlabel('Actual Value');

ylabel('Enhanced Value');

title('Transformation Curve');

%%%Create a 2D array for enhanced image%%%

Enhanced_HSV_Image=HSV_Image;

%%%Insert the enhanced value component%%%

Enhanced_HSV_Image(:,:,3)=uint8(Enhanced_Value);

%%%HSV to RGB colour space conversion%%%

Enhanced_RGB_Image = hsv2rgb(Enhanced_HSV_Image);

Contrast_Enhanced_Image=uint8(255*Enhanced_RGB_Image);