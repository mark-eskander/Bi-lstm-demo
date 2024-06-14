from fastapi import FastAPI
import uvicorn
import tensorflow
from utils import cleaned_tokenized

app = FastAPI( debug = True)



@app.get('/predict/{rev}' , status_code=200)
def predict(rev : str):
    rev= rev.replace('-',' ')
    model = tensorflow.keras.models.load_model(r'model_actual_june_demo2.h5')
    result= 1 if model.predict(cleaned_tokenized(rev)) > .5 else 0
    # (model.predict(cleaned_tokenized(rev)) > .5).astype(int)
    
    return {'prediction is' : result}

    # Run through terminal
if __name__== '__main__' :
    uvicorn.run(app,host='127.0.0.1',port=8000)
