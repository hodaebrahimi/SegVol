import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torch
from modelscope import snapshot_download
import numpy as np
import monai.transforms as transforms
from tqdm import tqdm
import nibabel as nib
join = os.path.join

####################################################################################

### SET PARAMS HERE ###
imagesTr_root  = '/uhome/hoda2/projects/p60290_1/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr'
cases_list     = './raos_tr_segvol/test_10cases.txt'
save_data_root = './raos_tr_segvol'
ckpt_path      = './segvol_test/SegVol_atlas11.pth'

# set device
gpu = 0
torch.cuda.set_device(gpu)

####################################################################################

os.makedirs(save_data_root, exist_ok=True)

# Segment these three classes separately.
# Key   = SegVol text prompt; Value = output filename stem.
category_reflect = {
    "colon":     "colon",
    "intestine": "small_bowel",   # SegVol calls the small bowel "intestine"
    "duodenum":  "duodenum",
}

# Load the list of case UIDs to process
with open(cases_list) as f:
    cases_names = [ln.strip() for ln in f if ln.strip()]
print(f'Detected {len(cases_names)} RAOS-Tr cases.')


class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)
        return d


class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
        return d


class RAOSDataset(Dataset):
    def __init__(self):
        self.images_root = imagesTr_root
        self.data        = cases_names
        self.img_loader  = transforms.LoadImage()
        self.transform4test = transforms.Compose([
            DimTranspose(keys=["image"]),
            MinMaxNormalization(),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.ToTensord(keys=["image"]),
        ])
        self.zoom_out_transform = transforms.Resized(
            keys=["image"], spatial_size=(32, 256, 256), mode='nearest'
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        case_name  = self.data[idx]
        image_path = join(self.images_root, f'{case_name}.nii.gz')

        ct_npy    = self.preprocess_ct(image_path)
        data_item = self.zoom_transform(ct_npy)
        data_item['case_num']   = case_name
        data_item['image_path'] = image_path
        return data_item

    def preprocess_ct(self, ct_path):
        ct_voxel_ndarray, _ = self.img_loader(ct_path)
        ct_voxel_ndarray = np.array(ct_voxel_ndarray).squeeze()
        ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray, axis=0)
        ct_voxel_ndarray = self.ForegroundNorm(ct_voxel_ndarray)
        return ct_voxel_ndarray

    def ForegroundNorm(self, ct_narray):
        flat = ct_narray.flatten()
        thred = np.mean(flat)
        voxel_filtered = flat[flat > thred]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 0.05)
        mean = np.mean(voxel_filtered)
        std  = np.std(voxel_filtered)
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - mean) / max(std, 1e-8)
        return ct_narray

    def zoom_transform(self, ct_npy):
        item = {'image': ct_npy}
        item = self.transform4test(item)
        item_zoom_out = self.zoom_out_transform(item)
        item['zoom_out_image'] = item_zoom_out['image']
        return item

    def save_pred(self, image_path, save_path, binary_mask,
                  start_coord, end_coord):
        """
        Save a binary mask (0/1 tensor, already thresholded) back into the
        original CT space. binary_mask must already be a clean binary tensor.
        """
        ct = nib.load(image_path)

        # Undo the axis swap applied during preprocessing
        sc = list(start_coord)
        ec = list(end_coord)
        sc[-1], sc[-3] = sc[-3], sc[-1]
        ec[-1], ec[-3] = ec[-3], ec[-1]

        # Undo axis swap in the mask itself
        binary_mask = binary_mask.transpose(-1, -3)

        preds_save = torch.zeros(ct.shape)
        preds_save[sc[0]:ec[0], sc[1]:ec[1], sc[2]:ec[2]] = binary_mask

        preds_np = preds_save.numpy().astype(np.uint8)
        nib.save(nib.Nifti1Image(preds_np, affine=ct.affine, header=ct.header), save_path)


####################################################################################

# Build model
model_dir = snapshot_download('yuxindu/SegVol')
clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, test_mode=True).cuda()
model.model.text_encoder.tokenizer = clip_tokenizer
model.eval()

model_dict = torch.load(ckpt_path)
model.load_state_dict(model_dict['model_state_dict'])
print('Model loaded.')

model = torch.nn.DataParallel(model, device_ids=[gpu])
dataset = RAOSDataset()
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

####################################################################################

for item in tqdm(test_loader):
    case_num   = item['case_num'][0]
    image_path = item['image_path'][0]

    case_save_dir = join(save_data_root, case_num)
    os.makedirs(case_save_dir, exist_ok=True)

    start_coord = list(item['foreground_start_coord'][0])
    end_coord   = list(item['foreground_end_coord'][0])

    # Run each class with its own text prompt and save a SEPARATE mask
    for prompt_cls, out_name in category_reflect.items():
        with torch.no_grad():
            logits_mask = model(
                image=item['image'].cuda(),
                zoomed_image=item['zoom_out_image'].cuda(),
                bbox_prompt_group=None,
                point_prompt_group=None,
                text_prompt=[prompt_cls],
                use_zoom=True,
            )
        pred_bin = (torch.sigmoid(logits_mask[0][0].cpu()) > 0.5).float()

        save_path = join(case_save_dir, f'{out_name}.nii.gz')
        dataset.save_pred(image_path, save_path, pred_bin, start_coord, end_coord)
        print(f'Saved: {save_path}')

print('Inference finished!')
