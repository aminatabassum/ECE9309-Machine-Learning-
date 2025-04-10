# ECE9309-Machine-Learning-

 Instructions to run swin, unetr, and Unet_CNN files  

1. SwinUnet
2. UnetR
   The final code is in the UNETR_cleaned.ipynb file. For the SwinUNet model, the input frame was upsampled to 128×128 pixels, but for UNetR, the original frame size of 112×112 is used. Since both SwinUNet and UNetR share the same workspace, to run the UNetR code, please replace the run_epoch() function in segmentation.py with the run_epoch() function provided in UNETR_cleaned.ipynb.
4. Unet_CNN
