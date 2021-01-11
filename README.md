# GDNN
A tensorflow implementation of Graph auto-encoders with deconvolutional networks. 

Details will come soon:
> Graph auto-encoders with deconvolutional networks.
> Google Research.

# Requirements
```
python            3.6.4
tensorflow        1.10
numpy             1.15.4
networkx          2.1
scipy             1.1.0
sklearn           0.20.1



```
# Set up the environments

pip install -r requirements.txt

# Dataset
DD,proteins,IMDB_M,IMDB_B,REDDIT_B

# Model options
```
  --epochs                      INT     Number of epochs.                  Default is 20.
  --learning-rate               FLOAT   Adam learning rate.                Default is 0.01.
 ```

# Example
```
train from scratch
```
python -W ignore main.py
