# datascience-contest
Codes for 1st Datascience Contest of KAIST-UNIST-POSTECH.

---------
### To start   
Create conda environment and install the required packages by the following command:   

```
conda create --name <env> --file requirements.txt
```

with the desired environment name in place of `<env>`.   
Then, upload the data `.csv` files under `datascience-contest/datasets`. Datasets are not disclosed to the public.   

### Training Models   
All the models (including the Knapsack Solver) are written in `model.py` file. The imported models are trained and evaluated in `train.py`. Simply run the following code:   

```
python train.py
```   

to obtain the results. You may use `--debug` argument to shorten the training process and check whether the codes work properly. Once the training process is completed, the parameters will be saved under `datascience-contest/saved_params/<current date and time>`. 
