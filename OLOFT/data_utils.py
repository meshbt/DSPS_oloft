import numpy as np
import random
from PIL import Image
import os
import pandas as pd

def process_train_val_split(train_images,train_labels,test_images,test_image_names,seed=999,target_folder='./data/processed'):

    random.seed(seed)
    np.random.seed(seed)

    val_idx = np.random.choice(len(train_images),int(len(train_images)*0.1),replace=False)
    train_idx = np.array([i for i in range(len(train_images)) if i not in val_idx])

    np.save(f'{target_folder}/train_labels.npy',train_labels[train_idx])
    np.save(f'{target_folder}/val_labels.npy',train_labels[val_idx])
    np.save(f'{target_folder}/test_image_names.npy',test_image_names)

    for i,j in enumerate(train_idx):
        Image.fromarray(train_images[j]).save(f'{target_folder}/train/{i}.jpg')

    for i,j in enumerate(val_idx):
        Image.fromarray(train_images[j]).save(f'{target_folder}/val/{i}.jpg')

    for i in range(len(test_images)):
        Image.fromarray(test_images[i]).save(f'{target_folder}/test/{i}.jpg')
    

def get_autogluon_datasets(target_folder='./data/processed'):
    train_labels = np.load(f'{target_folder}/train_labels.npy')
    val_labels = np.load(f'{target_folder}/val_labels.npy')

    clean_train_labels = np.where(train_labels<=0, 1, train_labels)
    clean_val_labels = np.where(val_labels<=0, 1, val_labels)

    train_path = os.path.abspath(f'{target_folder}/train')
    val_path = os.path.abspath(f'{target_folder}/val')
    test_path = os.path.abspath(f'{target_folder}/test')
    train_files = [f'{train_path}/{i}.jpg' for i in range(len(train_labels))]
    val_files = [f'{val_path}/{i}.jpg' for i in range(len(val_labels))]
    test_files = [f'{test_path}/{i}.jpg' for i in range(len(os.listdir(test_path)))]

    train_data = pd.DataFrame({'image': train_files, 'label': clean_train_labels})
    val_data = pd.DataFrame({'image': val_files, 'label': clean_val_labels})
    test_data = pd.DataFrame({'image': test_files})

    return train_data, val_data, test_data