from torchvision.transforms import v2
import torch 
import torch 
from PIL import Image

train_transforms = v2.Compose([
    v2.ColorJitter(brightness=0.3, contrast=0.3),
    v2.RandomAffine(degrees=0, shear=(-8, 8)),   # x-shear: slants, never clips
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
])

class OCR_dataset(Dataset):
  def __init__(self,image_dir,tsv_dir,transform=None):
      super().__init__()
      self.image_dir = image_dir
      self.tsv_dir =tsv_dir
      self.transform = transform
      self.tsv = pd.read_csv(tsv_dir,sep='\t', header=None,
                  names=['path', 'label', 'height', 'width'], quoting=3 ,na_filter=False,    # don't convert "NA"/"null"/"-"/"None" to NaN
    dtype={'label': str})
  def __len__(self):
    return len(self.tsv)
  def __getitem__(self,idx):
    img_id = self.tsv['path'].iloc[idx]
    img = os.path.join(self.image_dir,img_id)
    image =Image.open(img).convert('L')
    image = np.array(image)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image= image.to(torch.float32)
    image = image/255
    if self.transform is not None:
            image = self.transform(image)
    width = image.shape[-1]
    target = self.tsv['label'].iloc[idx]
    target_len  = len(target)
    target_idx = []
    for i in target:
      target_idx.append(chars_to_dict[i])
    target_idx = torch.tensor(target_idx)
    return image, target_len,target_idx,width

def make_collate():
    def collate_fn(batch):
        images, target_lens, target_idxs, widths = zip(*batch)

        max_width = max(widths)

        padded_images = []
        input_lengths = []

        for img, w in zip(images, widths):
            pad_amount = max_width - w
            padded = F.pad(img, pad=(0, pad_amount, 0, 0), mode='constant', value=0)
            padded_images.append(padded)
            input_lengths.append(w // 4)

        images_batch = torch.stack(padded_images)

        targets = torch.cat(target_idxs)


        target_lengths = torch.tensor(target_lens, dtype=torch.long)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        return images_batch, targets, input_lengths, target_lengths

    return collate_fn



