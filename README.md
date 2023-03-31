# EMST (An EfficientNet-based modified sigmoid transform for enhancing dermatological macro-images of melanoma and nevi skin lesions)
Matlab implemenation of An EfficientNet-based modified sigmoid transform for enhancing dermatological macro-images of melanoma and nevi skin lesions 

If any data or code is used, please cite our paper as,

Vipin Venugopal, Justin Joseph, M. Vipin Das, Malaya Kumar Nath, "An EfficientNet-based modified sigmoid transform for enhancing dermatological macro-images of melanoma and nevi skin lesions", Computer Methods and Programs in Biomedicine,Volume 222, 2022, 106935, https://doi.org/10.1016/j.cmpb.2022.106935.

# To run EMST Demo 
1. Download the Demo_EMST.zip file.
2. Unzip the file and give correct path location for test images.
3. The file contains (a) EMST_Pretarained_Model.m (b) Sig_Tran_Cross_Over Function.m (c) Demo_EMST_mlx.mlx (d) Demo_EMST.m (e) Test images (f) Ground-truth images


# To train the modified EfficientNet regressor
1. Use the V-component of the macro-images in HSV color space and the corresponding cross-over value given in the .csv files named Train and Test.
2. Aug1 to Aug9 folder contains the augmented versions of the V-component.
3. To download the dataset fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSeTJQxN80A3tb8UzWX5aJrXtcCLCFoOzLRLI073H31MVnlcTw/viewform) and contact us and we can provide a link to all the images.
4. Unzip the file and put all the folders and matlab files to the working directory.
5. We have split the dataset into train and test.
