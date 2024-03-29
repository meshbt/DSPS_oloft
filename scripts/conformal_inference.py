from OLOFT.inference_utils import get_conformal_threshold, get_conservative_predictions, save_json
from OLOFT.data_utils import get_autogluon_datasets
from autogluon.multimodal import MultiModalPredictor
import argparse
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from tqdm.autonotebook  import tqdm
import gdown
import uuid
import zipfile

args = argparse.ArgumentParser()
args.add_argument('--dataset_folder', type=str, default='./data/processed', help='Path to the folder including the preprocessed data including images generate. If you pass "online" the checkpoint will be downloaded. Default: ./data/processed')
args.add_argument('--model_folder', type=str, default='./models', help='Path to the folder to save the trained model. Default: ./models')
args.add_argument('--alpha', type=float, default=0.25, help='Alpha value to use for conformal prediction. Default: 0.25')
args.add_argument('--top_k', type=int, default=2, help='Number of top classes to consider for conservative prediction. Default: 2')
args.add_argument('--target_json', type=str, default='results.json', help='Path to the json file to save the results. Default: results.json')
args = args.parse_args()

# load the model
if args.model_folder == "online":
    if not os.path.exists("./checkpoint_1onLzn1LgTW80V_KX6AEgWY4J7blnWzho"):
        os.makedirs("./checkpoint_1onLzn1LgTW80V_KX6AEgWY4J7blnWzho")
    else:
        os.system('rm -rf ./checkpoint_1onLzn1LgTW80V_KX6AEgWY4J7blnWzho/*')
    gdown.download(id="1onLzn1LgTW80V_KX6AEgWY4J7blnWzho", output="./checkpoint_1onLzn1LgTW80V_KX6AEgWY4J7blnWzho/checkpoint.zip", quiet=False)
    with zipfile.ZipFile("./checkpoint_1onLzn1LgTW80V_KX6AEgWY4J7blnWzho/checkpoint.zip", 'r') as zip_ref:
        zip_ref.extractall("./checkpoint_1onLzn1LgTW80V_KX6AEgWY4J7blnWzho")
    
    predictor = MultiModalPredictor.load("./checkpoint_1onLzn1LgTW80V_KX6AEgWY4J7blnWzho")
else:
    predictor = MultiModalPredictor.load(args.model_folder)

# get the datasets
train_data, val_data, test_data = get_autogluon_datasets(target_folder=args.dataset_folder)
test_image_file_names = np.load(os.path.join(args.dataset_folder, 'test_image_names.npy'))

# get the conformal threshold
conformal_threshold = get_conformal_threshold(predictor, val_data, val_data['label'].to_numpy(), alpha=args.alpha)

# get the conservative predictions
conservative_predictions = get_conservative_predictions(predictor, test_data, conformal_threshold, top_k=args.top_k)

# save the results
save_json(test_image_file_names, conservative_predictions, target_json=args.target_json)