# Anomaly_detection_MultimodalUniverse_Data
Anomaly detection is an important task in astronomy. From a subset of this dataset, build a multimodal architecture that will detect anomalies in the data. Classify outliers vs regular data.

# 0. Create Environemnt & Installations

`conda create -n anomaly'

'git clone https://github.com/MultimodalUniverse/MultimodalUniverse.git'
 
 Install Required Libraries: Make sure you have all necessary libraries installed, including CLIP, DINO, and others for multimodal models.

`pip install -r requirements.txt`


# 1. Data
## Data Downloading 
Go to the link for the image dataset from the legacy survey: https://huggingface.co/datasets/MultimodalUniverse/legacysurvey

Go to hugging face , on "use this dataset" , copied 
in the terminal

`python3`

`from datasets import load_dataset`

`ds = load_dataset("MultimodalUniverse/legacysurvey")`

It will take many hours and then it will load the dataset. By default, the datasets library (from huggingface) stores datasets in a cache directory. You'll use the datasets library to load the data, filter it (if needed), and extract the images.

## Dataset Exploration:

Now we want to know the characteristics of this dataset.

`print(ds)`


It will give output like this :-

DatasetDict({
    train: Dataset({
        features: ['image', 'blobmodel', 'rgb', 'object_mask', 'catalog', 'EBV', 'FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4', 'SHAPE_R', 'SHAPE_E1', 'SHAPE_E2', 'object_id'],
        num_rows: 90580
    })
})

##  Subset Creation:

Since you want to focus on specific subsets of the data based on telescope and data type, you can filter the dataset accordingly. For example:
` subset = ds.filter(lambda example: example['telescope'] == 'your_telescope_name' and example['data_type'] == 'your_data_type')`

If this gives error then you need to upgrade your pillow library 

# 2. Feature Extraction:

You mentioned using models like CLIP, DINO, ResNet + VIT, or DEIT for feature extraction. For this, you'll need to preprocess the data (images, spectra, or light curves) and apply these models. You can use Hugging Face’s transformers library for models like CLIP and DINO, or you can use PyTorch for ResNet, VIT, and DEIT:
`python3 features_extraction.py`

# 3. Anomaly Detection:

    - CLIP/DINO: You can use these models for multimodal feature extraction (e.g., matching images with textual data). Use the CLIP model from Hugging Face if you have a torch higher than torch==1.7.1
    - ResNet + VIT + Isolation Forest: For anomaly detection, you can use Isolation Forests (which can be implemented via scikit-learn) on the features extracted from ResNet or VIT. Here’s an example using Isolation Forest:

    `python3 anomaly.py`
    


