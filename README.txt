This model was trained and evaluated on NIH HPC (Biowulf) cluster. 
The inference script accepts WSI images compatible with OpenSlide. 
The model was trained using FastAI library and inference also requires FastAI install

input = WSI image
output = geoJSON file compatible with QuPath

wsi_reqs.txt has all package versions from conda env. 
The following steps for miniconda env set-up:
	conda create -n wsi_zhao python=3.8 numpy scipy
	conda activate wsi_zhao
	conda install -c bioconda openslide
	conda install -c bioconda openslide-python
	pip install Shapely
	pip install geojson
	conda install -c conda-forge opencv
	conda install pandas
	conda install -c fastchan fastai
	pip install semtorch

Example inference:
python -W ignore wsi_tumorsegment_simple.py --file_location "./example_images" --image_file "example.ndpi" --save_name "EX1" --save_dir "./example_output"


A docker container that can be used as web-based interface for users to upload images and download geojson results is available. Further documentation is coming
docker pull curtislisle/air_adenocarcinoma_v1_11_2021
