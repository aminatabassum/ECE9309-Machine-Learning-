import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class EchoNetDataset(Dataset): # this would be my EchoNet dataset instead
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("_img_","_mask_"))
        # print(img_path)
        image = np.array(Image.open(img_path).convert("RGB")) #using numby because PIL wants numpy (augmentations library)
        mask = np.array(Image.open(mask_path).convert("RGBA"))  # Load as RGBA
        alpha_channel = mask[:, :, 3]  # Extract alpha channel
        # mask = np.array(alpha_channel, dtype=np.float32)  # Convert to float32

        # Convert the alpha channel to a binary mask (0 for transparent, 1 for opaque)
        mask = np.where(alpha_channel == 0, 1.0, 0.0).astype(np.float32)  #transparent is the object
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # float: 0.0, 255.0
        #some preprocessing

        # mask[mask==255.0] = 1.0 # we are going to use sigmoid for last activation. explain why 1 is better
#maskAlt has opaque regions == 255, while masks is suppose to check for values == 0 and set them to 1
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    