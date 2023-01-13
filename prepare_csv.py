import os
import json
import pandas as pd


#results_dir = "/netscratch/agautam/experiments/logs/evaluation"
results_dir = "/netscratch/agautam/experiments/logs/evaluation_freeze"

rows = []

nice_model_names = {
    'malteos/gpt2-wechsel-german-ds-meg': 'gpt2-wechsel-german',
    'bert-base-german-cased': 'bert-german',
    'malteos/bloom-1b7-twc-german': 'bloom-1b7-german',
    'malteos/gpt2-xl-wechsel-german': 'gpt2-xl-german'
}

nice_dataset_names = {
    'philschmid/germeval18': 'germeval18',
    'deepset/germanquad' : 'german-quad',
    'akash418/germeval_2017': 'germeval17',
    'elenanereiss/german-ler': 'german-ler',
    'gnad10': 'gnad10',
    'akash418/german_europarl': 'german-europarl'
}

for fn in os.listdir(results_dir):
    with open(os.path.join(results_dir, fn)) as f:
        res = json.load(f)
        print(res)
        '''
        if(res['problem_type'] == 'classification' or res['problem_type'] == 'ner'):
            row = {
                'model_name': res['model_name'],
                'dataset_name': res['dataset_name'],
                'eval_accuracy': res['eval_accuracy'],
                'problem_type': res['problem_type'],
                'eval_samples': res['eval_samples']
            }
        '''
        res['model_name'] = nice_model_names[str(res['model_name'])]
        res['dataset_name'] = nice_dataset_names[str(res['dataset_name'])]
        row = {
            'dataset_name': res['dataset_name']
        }
        # dump everything to dataframe
        row.update(res)
        rows.append(row)

df = pd.DataFrame(rows)
#df.to_csv('results_data.csv', index = False)
df.to_csv('results_data_freeze.csv', index = False)