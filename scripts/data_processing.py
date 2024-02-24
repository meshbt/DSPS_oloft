from OLOFT.data_utils import process_train_val_split
import argparse
import os
import numpy as np
import gdown

args = argparse.ArgumentParser()
args.add_argument('--seed', type=int, default=999, help='Random seed used for train/val split. Default: 999')
args.add_argument('--dataset_folder', type=str, default='./data', help='Path to the folder including the preprocessed dataset in npy files, Default: ./data')
args.add_argument('--resolution', type=int, default=512, help='Resolution of the images to use to create the dataset. One of 64,128,256,512. Default: 512')
args = args.parse_args()

gd_links = {
    64: {
        'train_images': '1S7SXVrvq4vedpAamTCuwgjriSe1Ymj_O',
        'test_images': '1F6d2An9dGkn8xK2c61sXexdyaEG4mEpb'
    },
    128: {
        'train_images': '13lageq-i-f-BjoLjuK7bnRNd6Vt6LLqi',
        'test_images': '1vwGQ3cNsqsCbEW8N-cktadDnx2wyr1hB'
    },
    256: {
        'train_images': '1gSCH5SYsrAVCpFSWKoTMG3QF2kkEkAoO',
        'test_images': '1XE8H5Pecd5sHPqlPMMy7pMPVZZ7QIhij'
    },
    512: {
        'train_images': '1E2xiuhgdvtBs8Wd6GR5HVyRz6qjVzoKb',
        'test_images': '1iv06AB1nSJovnTr1DlNyINLFdeH_xhqJ'
    },
    "train_labels": '1SV2WTWMFXaYALAe_-9Q0rogBa_UfIsRR',
    "test_image_names": '1F4zB0VLtOOr1FjN_AO8x9FhiY25JJdkQ'
}

# check is dataset folder exists
if not os.path.exists(args.dataset_folder):
    os.mkdir(args.dataset_folder)

if not os.path.exists(os.path.join(args.dataset_folder, 'processed/')):
    os.mkdir(os.path.join(args.dataset_folder, 'processed/'))

if not os.path.exists(os.path.join(args.dataset_folder, 'processed/train/')):
    os.mkdir(os.path.join(args.dataset_folder, 'processed/train/'))

if not os.path.exists(os.path.join(args.dataset_folder, 'processed/val/')):
    os.mkdir(os.path.join(args.dataset_folder, 'processed/val/'))

if not os.path.exists(os.path.join(args.dataset_folder, 'processed/test/')):
    os.mkdir(os.path.join(args.dataset_folder, 'processed/test/'))

# empty the folders
for folder in ['train', 'val', 'test']:
    for file in os.listdir(os.path.join(args.dataset_folder, 'processed', folder)):
        os.remove(os.path.join(args.dataset_folder, 'processed', folder, file))

if os.path.exists(os.path.join(args.dataset_folder, 'processed', 'test_image_names.npy')):
    os.remove(os.path.join(args.dataset_folder, 'processed', 'test_image_names.npy'))

if os.path.exists(os.path.join(args.dataset_folder, 'processed', 'train_labels.npy')):
    os.remove(os.path.join(args.dataset_folder, 'processed', 'train_labels.npy'))

if os.path.exists(os.path.join(args.dataset_folder, 'processed', 'val_labels.npy')):
    os.remove(os.path.join(args.dataset_folder, 'processed', 'val_labels.npy'))

# check if the dataset files exist and download them if not
if not os.path.exists(os.path.join(args.dataset_folder, f'train_images_{args.resolution}.npy')):
    gdown.download(id=gd_links[args.resolution]['train_images'], output=os.path.join(args.dataset_folder, f'train_images_{args.resolution}.npy'), quiet=False)

if not os.path.exists(os.path.join(args.dataset_folder, f'test_images_{args.resolution}.npy')):
    gdown.download(id=gd_links[args.resolution]['test_images'], output=os.path.join(args.dataset_folder, f'test_images_{args.resolution}.npy'), quiet=False)

if not os.path.exists(os.path.join(args.dataset_folder, 'train_labels.npy')):
    gdown.download(id=gd_links['train_labels'], output=os.path.join(args.dataset_folder, 'train_labels.npy'), quiet=False)

if not os.path.exists(os.path.join(args.dataset_folder, 'test_image_names.npy')):
    gdown.download(id=gd_links['test_image_names'], output=os.path.join(args.dataset_folder, 'test_image_names.npy'), quiet=False)

# load the dataset
train_images = np.load(os.path.join(args.dataset_folder, f'train_images_{args.resolution}.npy'))
train_labels = np.load(os.path.join(args.dataset_folder, 'train_labels.npy'))
train_labels = np.where(train_labels<=0, 1, train_labels)
test_images = np.load(os.path.join(args.dataset_folder, f'test_images_{args.resolution}.npy'))
test_image_names = np.load(os.path.join(args.dataset_folder, 'test_image_names.npy'))

# process the train/val split
process_train_val_split(train_images, train_labels, test_images, test_image_names, seed=args.seed, target_folder=os.path.join(args.dataset_folder, 'processed'))

print('The images for training/validation/test have been processed and saved in the folder: ', os.path.join(args.dataset_folder, 'processed'))
