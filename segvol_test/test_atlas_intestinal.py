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
imagesTr_root  = '/uhome/hoda2/projects/p60290_1/AbdomenAtlas1.1Dataset'
labelsTr_root  = '/uhome/hoda2/projects/p60290_1/AbdomenAtlas1.1Dataset'  # GT segmentations are inside each case folder
save_data_root = './segvol_test/Atlas_Intestinal_Predict'
ckpt_path      = './segvol_test/SegVol_atlas11.pth'

# set device
gpu = 0
torch.cuda.set_device(gpu)

####################################################################################

os.makedirs(save_data_root, exist_ok=True)

# Only segment these three classes
category_reflect = {
    "colon":     "colon",
    "duodenum":  "duodenum",
    "intestine": "intestine",
}

# Load AbdomenAtlas cases matching BDMAP_* directories
all_files = sorted(os.listdir(imagesTr_root))
cases_names = [f for f in all_files if 'BDMAP_' in f and '.' not in f]
print(f'Detected {len(cases_names)} AbdomenAtlas cases.')


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


class IBDDataset(Dataset):
    def __init__(self):
        self.images_root = imagesTr_root
        self.labels_root = labelsTr_root
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
        case_name = self.data[idx]
        image_path = join(self.images_root, case_name, 'ct.nii.gz')

        ct_npy    = self.preprocess_ct(image_path)
        data_item = self.zoom_transform(ct_npy)
        data_item['case_num']   = case_name
        data_item['image_path'] = image_path   # pass along so save_preds can use it

        if self.labels_root is not None:
            # Build combined GT from colon + duodenum + intestine segmentations
            seg_dir = join(self.labels_root, case_name, 'segmentations')
            combined = None
            for organ in category_reflect.keys():
                seg_path = join(seg_dir, f'{organ}.nii.gz')
                if os.path.exists(seg_path):
                    seg_data, _ = self.img_loader(seg_path)
                    seg_data = np.array(seg_data).squeeze()
                    if combined is None:
                        combined = seg_data
                    else:
                        combined = np.maximum(combined, seg_data)
            if combined is not None:
                data_item['label'] = combined
            else:
                data_item['label'] = np.array([])
        else:
            data_item['label'] = np.array([])

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

    def save_merged_pred(self, image_path, save_path, merged_binary_mask,
                         start_coord, end_coord):
        """
        Save a pre-computed binary mask (0/1 tensor, already thresholded)
        back into the original CT space.

        NOTE: merged_binary_mask must already be a clean binary tensor —
        do NOT apply sigmoid again here.
        """
        ct = nib.load(image_path)

        # Undo the axis swap applied during preprocessing
        sc = list(start_coord)
        ec = list(end_coord)
        sc[-1], sc[-3] = sc[-3], sc[-1]
        ec[-1], ec[-3] = ec[-3], ec[-1]

        # Undo axis swap in the mask itself
        merged_binary_mask = merged_binary_mask.transpose(-1, -3)

        preds_save = torch.zeros(ct.shape)
        preds_save[sc[0]:ec[0], sc[1]:ec[1], sc[2]:ec[2]] = merged_binary_mask

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
dataset = IBDDataset()
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

####################################################################################

for item in tqdm(test_loader):
    case_num   = item['case_num'][0]
    image_path = item['image_path'][0]

    case_save_dir = join(save_data_root, case_num)
    os.makedirs(case_save_dir, exist_ok=True)

    # Collect raw logits for each class (keep on CPU to save GPU memory)
    pred_logits = []
    for default_cls, prompt_cls in category_reflect.items():
        with torch.no_grad():
            logits_mask = model(
                image=item['image'].cuda(),
                zoomed_image=item['zoom_out_image'].cuda(),
                bbox_prompt_group=None,
                point_prompt_group=None,
                text_prompt=[prompt_cls],
                use_zoom=True,
            )
        pred_logits.append(logits_mask[0][0].cpu())

    # Post-process: sigmoid → threshold → union across three classes
    pred_stack   = torch.stack(pred_logits, dim=0)          # (3, D, H, W)
    pred_bin     = (torch.sigmoid(pred_stack) > 0.5).float()
    merged_mask  = (pred_bin.sum(dim=0) > 0).float()        # (D, H, W), binary

    # Save — pass the already-binary merged_mask; save_merged_pred does NOT sigmoid again
    merged_save_path = join(case_save_dir, f'{case_num}_pred_binary.nii.gz')
    dataset.save_merged_pred(
        image_path, merged_save_path, merged_mask,
        start_coord=list(item['foreground_start_coord'][0]),
        end_coord=list(item['foreground_end_coord'][0]),
    )
    print(f'Saved: {merged_save_path}')

print('Inference finished!')
