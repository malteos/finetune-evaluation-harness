# Finetune-evaluation-harness


## Docker

To run all scripts within our Slurm cluster, everything needs to be containerized with Docker.

Image:
- Name: `malteos/finetune-eval` (available at: https://hub.docker.com/repository/docker/malteos/finetune-eval )

CI/CD:
- We use GitHub actions to automatically build and push new Docker images (see `.github` directory)
- To trigger a new build just include the string `docker build` in your commit message. Example commit message `git commit -am "updated dependencies (docker build)"`
- Building a new image is  **only required when installing/changing the Python packages!**
    For code changes a `git pull` is so sufficient since this repo is mounted into the container.


## Slurm

Import latest Docker image as enroot:
```bash
srun enroot import -o /netscratch/$USER/enroot/malteos+finetune-eval+latest.sqsh docker://malteos/finetune-eval:latest
```

## Usage
- This repo contains seperate script for each of the tasks until the master script is ready.

- Here, we are sharing some examples and certain parameters to be added to use the script in the expected way. All the credits for the orginal development go to Huggingface contributers, please regularly monitor the requirements file incase of some dependency update https://github.com/huggingface/transformers/tree/main/examples/pytorch

- If you fail to understand what any of the paramater does, --help is your friend. Other keyword arguments used for HF transformers library are mentioned here: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

- Sample example for using main master script for running the tasks

```
python main.py --model bert-base-german-cased --task_list germeval2018 germeval2017 gnad10
```
Additional parameters can be specified for the master script.

### Datasets Used and HuggingFace Id's
1. philschmid/germeval18 -- GERMEVAL 18
2. elenanereiss/german-ler -- GERMAN_NER_LEGAL
3. gnad10 -- GNAD10
4. deepset/germanquad -- GERMAN_QUAD
5. akash418/germeval_2017 -- GERMEVAL_2017
6. akash418/german_europarl -- GERMAN EUROPARL


Incase you want to have a more fine-grained control over the scripts, the use the examples given below

### Text Classification Task
```
python hgf_fine_tune_class.py \
--model_name_or_path malteos/gpt2-wechsel-german-ds-meg --dataset_name philschmid/germeval18 \
--do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 16 --num_train_epochs 1 \
--output_dir /netscratch/agautam/experiments/test_logs/sample --overwrite_output_dir True \
--label_value binary --ignore_mismatched_sizes --remove_labels multi
```
- label_value is required to identify the column on which text classification is required
- remove_labels will exclude the columns which are not needed. Note: this behaviour is not by default, so you will have to check the dataset columns and
specify the ones which you dont want, it becomes all the more important in multi-label class.
- do_eval makes sure that evaluation metrics are run and saved as json on the validation or the test split (which ever is avaialble on HF). Many HF datasets do not naturally have the test split.
- Incase for some dataset (eg GERMEVAL 2017), an error is thrown saying: ``` TypeError: TextInputSequence must be str ```, then try and add a parameter --use_fast_tokenizer False


### Token Classfication Task
- The HF developers provided a common script for ner tagging, pos tagging and chunk tag identification. Hence it becomes all the more important to be sure of the exact task you want to run, the dataset format and how it was created.
-  HF expects the dataset for these task to be of certain type, for reference please check out ``` elenanereiss/german-ler ``` dataset hosted on HF. If you want to run the your own dataset using this script, the best way would be to model the data in the exact same format. Take special care about <b>column type</b> and the name of the column. The default task is ner identification and it expects the column name similar to ```ner_tags``` to exist in the dataset for this task.
- Another thing to note is the importance of having feature file. Certain datasets hosted on hub with stringent requiremets (pre 2021) have feature file for each the token classification task. A feature label contains the mapping between various types of tags. Also it differenciates between B- tag and I- tag, if it exists. Read more, here on what it is: https://stackoverflow.com/questions/53933854/what-is-the-list-of-possible-tags-with-a-description-of-conll-2003-ner-task

- Here, we are sharing two examples for this task. One for dataset hosted on hub, in the expected format and the other one which was modifed in the expected format and saved as json as train and test file

```
 python hgf_fine_tune_ner.py  --model_name_or_path malteos/gpt2-wechsel-german-ds-meg \
 --dataset_name elenanereiss/german-ler --output_dir /netscratch/agautam/experiments/test_logs/sample \
 --do_train --do_eval  --overwrite_output_dir --num_train_epochs 1 --feature_file True --max_seq_length 512
```
-Note the special argument --feature_file True, this ensures that the version of dataset hosted on HF has a feature file.

```
python hgf_fine_tune_ner.py  --model_name_or_path malteos/gpt2-wechsel-german-ds-meg \
--train_file /netscratch/agautam/experiments/local_dataset/europarl/train.json \
--validation_file /netscratch/agautam/experiments/local_dataset/europarl/test.json \
--output_dir /netscratch/agautam/experiments/test_logs/sample --do_train --do_eval \
--task_name ner --label_column_name ner_tags \
--text_column_name tokens \
--overwrite_output_dir --num_train_epochs 1 --feature_file False

```
- This is the script to run ner identification on german europral dataset saved in a local directory as a json file. Note here we have to specify task_name ner and label_column_name and even the column containing the text column name. Lastly the boolean param --feature_file set to False.

### Question Answering

```
python hgf_fine_tune_qa.py  --model_name_or_path bert-base-german-cased \
--dataset_name deepset/germanquad \
--output_dir /netscratch/agautam/experiments/test_logs/sample \
--do_train --do_eval  --overwrite_output_dir --num_train_epochs 1 --max_seq_length 512
```

- Default script and parameters, no major additions to the script. Only thing to note is to be careful if the model which you want to run has a an AutoModel For Question Answering Type else it throws an error.
