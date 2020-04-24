FROM hiromuhota/fonduer:0.8.1

RUN pip install mlflow==1.8.0
RUN python -m spacy download en
