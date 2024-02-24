# DSPS Submission For Team Kirk O'Loft
This repo includes all the code needed to replicate the results. The required packages and version used are included in the `requirements.txt` file.

## Training & Testing
The code for training and conformal inference is included in the scripts folder with utilites defined in the OLOFT folder. To run everything as we did and replicate the results please run OLOFT.ipynb. To get the exact results make sure the versions of packages you have match ours. Use a GPU districbution of pytorch 2.

### Data
We pre-process the data into standard resolutions. This data will be automatically downloaded for you when you run the scripts. Alternatively, you can grab the npy files needed from [here](https://drive.google.com/drive/folders/1XnKkrRMxCykbFGbu-J0Prab4V8SdfWm2?usp=drive_link) and place them in the data folder.

### Checkpoints
If versions of packages are the same and the seeds set in OLOFT.ipynb are kept the results should be exactly replicated. However the checkpoint folder contents can be downloaded as a zip file [here](https://drive.google.com/file/d/1onLzn1LgTW80V_KX6AEgWY4J7blnWzho/view?usp=sharing). Extract these files to the checkpoint folder.