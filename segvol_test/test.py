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
test_data_root = 'path/to/AbdomenAtlasTest'
save_data_root = 'path/to/AbdomenAtlasPredict'
ckpt_path = './SegVol_atlas11.pth'

# set device
gpu = 0
torch.cuda.set_device(gpu)

####################################################################################

os.makedirs(save_data_root, exist_ok=True)

# categories(default names to segvol names)
category_reflect = {
    "aorta": "aorta",
    "gall_bladder": "gall_bladder",
    "kidney_left": "left kidney",
    "kidney_right": "right kidney",
    "liver": "liver",
    "pancreas": "pancreas",
    "postcava": "postcava",
    "spleen": "spleen",
    "stomach": "stomach",
    "adrenal_gland_left": "adrenal_gland_left",
    "adrenal_gland_right": "adrenal_gland_right",
    "bladder": "bladder",
    "celiac_trunk": "celiac_trunk",
    "colon": "colon",
    "duodenum": "duodenum",
    "esophagus": "esophagus",
    "femur_left": "femur_left",
    "femur_right": "femur_right",
    "hepatic_vessel": "hepatic_vessel",
    "intestine": "intestine",
    "lung_left": "left lung",
    "lung_right": "right lung",
    "portal_vein_and_splenic_vein": "portal_vein_and_splenic_vein",
    "prostate": "prostate",
    "rectum": "rectum"
  }

# test data load
cases_names = os.listdir(test_data_root)
cases_names = [x for x in cases_names if 'BDMAP_' in x and '.' not in x]
print(f'Detect {len(cases_names)} test cases.')

# dataset
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
    
class AtlasDataset(Dataset):
    def __init__(self):
        self.root = test_data_root
        self.category_reflect = category_reflect
        self.data = cases_names
        #### func
        self.img_loader = transforms.LoadImage()
        self.transform4test = transforms.Compose(
            [
                DimTranspose(keys=["image"]),
                MinMaxNormalization(),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                transforms.ToTensord(keys=["image"]),
            ]
        )
        self.zoom_out_transform = transforms.Resized(keys=["image"], spatial_size=(32,256,256), mode='nearest')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        case_name = self.data[idx]
        ct_path = join(test_data_root, case_name, 'ct.nii.gz')
        ct_npy = self.preprocess_ct(ct_path)
        data_item = self.zoom_transform(ct_npy)
        data_item['case_name'] = case_name
        return data_item
    
    def preprocess_ct(self, ct_path):
        # generate ct_voxel_ndarray
        ct_voxel_ndarray, _ = self.img_loader(ct_path)
        ct_voxel_ndarray = np.array(ct_voxel_ndarray).squeeze()
        ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray, axis=0)
        ct_voxel_ndarray = self.ForegroundNorm(ct_voxel_ndarray)
        return ct_voxel_ndarray
    
    def ForegroundNorm(self, ct_narray):
        ct_voxel_ndarray = ct_narray.copy()
        ct_voxel_ndarray = ct_voxel_ndarray.flatten()
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 00.05)
        mean = np.mean(voxel_filtered)
        std = np.std(voxel_filtered)
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - mean) / max(std, 1e-8)
        return ct_narray
    
    def zoom_transform(self, ct_npy):
        item = {
            'image': ct_npy,
        }
        item = self.transform4test(item)
        item_zoom_out = self.zoom_out_transform(item)
        item['zoom_out_image'] = item_zoom_out['image']
        return item
    
    def save_preds(self, case_name, save_path, logits_mask, start_coord, end_coord):
        ct_path = join(test_data_root, case_name, 'ct.nii.gz')
        ct = nib.load(ct_path)
        logits_mask = logits_mask.transpose(-1, -3)
        start_coord[-1], start_coord[-3] = start_coord[-3], start_coord[-1]
        end_coord[-1], end_coord[-3] = end_coord[-3], end_coord[-1]
        preds_save = torch.zeros(ct.shape)
        preds_save[start_coord[0]:end_coord[0], 
                        start_coord[1]:end_coord[1], 
                        start_coord[2]:end_coord[2]] = torch.sigmoid(logits_mask)
        preds_save = torch.where(preds_save > 0.5, 1., 0.).numpy()
        preds_nii = nib.Nifti1Image(preds_save, affine=ct.affine, header=ct.header)
        nib.save(preds_nii, save_path)

# build model
model_dir = snapshot_download('yuxindu/SegVol')
clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, test_mode=True).cuda()
model.model.text_encoder.tokenizer = clip_tokenizer
model.eval()

# resume
model_dict = torch.load(ckpt_path)
model.load_state_dict(model_dict['model_state_dict'])
print('model load done')

model = torch.nn.DataParallel(model, device_ids=[gpu])
dataset = AtlasDataset()
test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

# begin test
for item in tqdm(test_loader):
    case_name = item['case_name'][0]
    case_save_path = join(save_data_root, case_name, 'predictions')
    os.makedirs(case_save_path, exist_ok=True)
    # loop for categories
    for default_cls, prompt_cls in category_reflect.items():
        with torch.no_grad():
            logits_mask = model(image=item['image'].cuda(),
                zoomed_image=item['zoom_out_image'].cuda(),
                bbox_prompt_group=None,
                point_prompt_group=None,
                text_prompt=[prompt_cls],
                use_zoom=True
                )
        case_cls_save_path = join(case_save_path, default_cls+'.nii.gz')
        dataset.save_preds(case_name, case_cls_save_path, logits_mask[0][0], 
                           start_coord=list(item['foreground_start_coord'][0]), 
                           end_coord=list(item['foreground_end_coord'][0]))
        print(case_cls_save_path, ' save done')

print('Test finished!')