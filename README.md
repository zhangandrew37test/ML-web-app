# Machine Learning App

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]()

# Reproducing this web app

To recreate this web app on your own computer, do the following.

### Create conda environment

Firstly, we will create a conda environment called _ml_

```
conda create -n ml python=3.7.9
```

Secondly, we will login to the _ml_ environement

```
conda activate ml
```

### Install prerequisite libraries

Download requirements.txt file

```

wget https://raw.githubusercontent.com/zhangandrew37/ML-web-app/main/requirements.txt?token=ANATUZX3NACXFJU3GI6WOELBF6WX6

```

Pip install libraries

```
pip install -r requirements.txt
```

### Download and unzip contents from GitHub repo

<!--- change below -->

Download and unzip contents from https://github.com/zhangandrew37/ML-web-app/archive/main.zip

### Launch the app

```
streamlit run app.py
```
