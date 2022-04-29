# Implicit GPS-based bicycle route choice model

Code used for our paper [Implicit GPS-based bicycle route choice model using clustering methods and a LSTM network](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0264196).


## Dependencies and virtual environment

We recommend the use of a virtual python environment. The notebooks are configured to work with a virtual environment named `repo_paper`. Use the package manager [pip](https://pypi.org/project/pip/) to install the dependencies.

```bash
pip install pipenv
pipenv install ipykernel
pipenv shell
python -m ipykernel install --user --name=repo_paper
pip install -r requirements.txt
jupyter notebook
```
In your notebook, Kernel -> Change Kernel. `repo_paper` should be an option.

## Usage

The code is configured for `monresovelo` and has already been executed for it, which means that all the files related to this project (clusters, neural network, graphs, etc.) are already created. To restart the creation of these files, delete `files/monresovelo/` and put the `json` file of the MonResoVelo gps tracks (accessible [here](https://open.canada.ca/data/en/dataset/77f30d2b-c786-45f0-9f33-ebdef46f3b4c)) in `data/monresovelo/`. The project have to be executed in this order :
1. `data_creation.ipynb` : preprocess the data found in `data/moresovelo`, load and save the road graph as well as the shortest paths and the Mapbox routes.

2. `main.ipynb` : shows the data graphically through maps and graphs, cleans the preprocessed data and clusters the observations as well as the voxels.

3. `python/main_NN.py` : trains a LSTM neural network.

4. `main.py` : uses all the previously created files to applicate our method and generates the graphs that we show as results in the paper.

Warning : When generating the mapbox routes, `data_creation.ipynb` needs a token to access the Mapbox API. Create a free account on [their website](https://www.mapbox.com/) and replace the `token` variable in the notebook by your token.

## Using your own data

If you want to execute the code for other data than `monresovelo`, you have to :
1. Create the directory `data/PROJECT_NAME/` and put your data in it.
2. Create your own `data/PROJECT_NAME/load_data.py` script by taking the example of `data/monresovelo/load_data.py`.
3. Copy `data/monreseovelo/load_city.py` in `data/PROJECT_NAME/`.
4. Execute the project by modifying the `project_folder` variables in the notebooks and specifying the project name in the arguments of the python files.

