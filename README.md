## Create a virtual env
Sandboxing all Python development in a virtual environment is a recommended practice.
This will ensure any Python packages you install for this application will not affect or conflict with the system-level packages.
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
Because if there's no deep learning framework then what are we even doing here?
```
pip install tensorflow tensorflow_hub
```

## Start retraining
This may take anywhere between 10-40 mins.
My MacBook Pro took 12 mins, while my Windows laptop took 37 mins.
```
python retrain.py --image_dir celeb-photos --logdir logs --saved_model_dir=saved-model
```

## Test inference
```
python label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=bill.jpg
```
## Install Flask and friends
Required for creating an HTTP API for inferences.
After all, we are developing for normal people (who use websites) rather than geeks (who use command-line).
```
pip install flask Flask-Uploads
```

## Start the server
This will host both our image classification API and frontend webpage that we'll use to test our API.
```
python server.py
```

## Test the API in Postman
```
POST /api/infer HTTP/1.1
Host: 127.0.0.1:5000
Cache-Control: no-cache
Postman-Token: 3e108526-8c08-ea9c-393b-ce8b58ee66fb
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="target"; filename="bill.jpg"
Content-Type: image/jpeg


------WebKitFormBoundary7MA4YWxkTrZu0gW--
```
![Testing the API in Postman](docs/Testing-the-API-in-Postman.png)

## Test the API from webpage
![Using the website](docs/Celebrity-Recognizer-Website.png)