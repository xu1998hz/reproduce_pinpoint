import argparse 
from comet import download_model, load_from_checkpoint
import pickle
import json
# remember to switch to 'comet' environment

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="Unbabel/wmt22-comet-da")
argparser.add_argument("--data", type=str, default="../out/data/zh-en_wmt_test_wmt23_out_clean.pkl")
argparser.add_argument("--out_path", type=str, default="../out/comet_zh-en_wmt_test_wmt23.json")
args = argparser.parse_args()

# Data must be in the following format:
# data = [
#     {
#         "src": "10 到 15 分钟可以送到吗",
#         "mt": "Can I receive my food in 10 to 15 minutes?",
#         "ref": "Can it be delivered between 10 to 15 minutes?"
#     },
#     {
#         "src": "Pode ser entregue dentro de 10 a 15 minutos?",
#         "mt": "Can you send it for 10 to 15 minutes?",
#         "ref": "Can it be delivered between 10 to 15 minutes?"
#     }
# ]
data = pickle.load(open(args.data, 'rb'))


model_path = download_model(args.model)

# Load the model checkpoint:
model = load_from_checkpoint(model_path)

# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1)
# print(model_output)
# print(model_output.scores) # sentence-level scores
print(f'comet score: {model_output.system_score}') # system-level score

data_dict = {'src': [], 'out': [], 'seg': []}
with open(args.out_path, 'w') as f:
    data_dict['sys'] = model_output.system_score
    for i in range(len(data)):
        data_dict['seg'].append(model_output.scores[i])
        data_dict['src'].append(data[i]['src'])
        data_dict['out'].append(data[i]['mt'])
    json.dump(data_dict, f)

# Not all COMET models return metadata with detected errors.
# print(model_output.metadata.error_spans) # detected error spans