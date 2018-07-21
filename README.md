## Create a virtual env
```
pip install --upgrade virtualenv
virtualenv --system-site-packages tensorflow
cd tensorflow
source ./bin/activate
```

To deactivate later:
```
deactivate
```

## Install Tensorflow and friends
```
pip install tensorflow
```

## Start retraining
```
python retrain.py --image_dir celeb-photos --logdir logs --saved_model_dir=saved-model
```

## Test inference
```
python label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=bill.jpg
```
## Install Flask and friends
```
pip install flask Flask-Uploads
```

## Host the API
```
python api.py
```
