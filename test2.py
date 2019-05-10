import pandas as pd
import sys
import mlflow.pyfunc

model_path = "fonduer_model"
loaded_model = mlflow.pyfunc.load_pyfunc(model_path)
filename = sys.argv[1]
model_input = pd.DataFrame({'filename': [filename]})
df = loaded_model.predict(model_input)
print(df)