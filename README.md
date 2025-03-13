# FashionMnist-Classifier

Running the project:
1) clone the repository to a local project folder
```bash
git clone https://github.com/Devadeut/FashionMnist-Classifier.git
cd FashionMnist-Classifier
```
2) Download the model ("best_model.pth") to the project folder from the following link-
```
https://drive.google.com/file/d/1-1xR16R0LWE5a65bq-9QsSq_5iqA3Fuh/view?usp=sharing
```
To train a new model instead, refer to jupyter notebook "model.ipynb"
3.1) To use the FASTAPI, run following commands in the local project folder
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
3.2) To use DOCKER instead, run
```bash
docker build -t fastapi-app .
docker run -p 8000:8000 fastapi-app
```
The API can then be tested by visiting "http://127.0.0.1:8000/docs" in both cases.

NOTE: As this is just a barebone implementation, nothing will be displayed on "http://127.0.0.1:8000". Go to "http://127.0.0.1:8000/docs" to test the '/predict' endpoint. When prompted with username and password, use 'user' and 'pass' respectively.
We provide a sample image but any other images from FashionMNNIST dataset can also be used.

