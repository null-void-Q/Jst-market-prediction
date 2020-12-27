#import janestreet
import pandas as pd
import torch
from nullnet import NullNet
from services import predict

MODEL_PATH = './x0_model_ex52.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NullNet()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # to disable dropout and normalization and such
model.to(device)


data = pd.read_csv('../data/example_test.csv',dtype='float32')
features = data.iloc[:,data.columns.str.contains('feature')]
means = features.mean()
features = features.fillna(means)
inputs = torch.tensor(features.values,dtype=torch.float32)
predictions = predict(model,inputs,device)
print(predictions.cpu().data.numpy())

#env = janestreet.make_env() # initialize the environment
#iter_test = env.iter_test() # an iterator which loops over the test set

# for (test_df, prediction_df) in iter_test:
#     X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
#     preds = model(X_test)
#     prediction_df.action = 0 #make your 0/1 prediction here
#     env.predict(prediction_df)
