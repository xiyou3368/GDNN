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

# Dataset
Default is IMDB_B, for other datasets, please refer to the branches, e.g., DD,proteins,IMDB_M,,REDDIT_B

# Model options
```
  --epochs                      INT     Number of epochs.                  Default is 20.
  --learning-rate               FLOAT   Adam learning rate.                Default is 0.01.
 ```

# Example

train from scratch
```
python -W ignore main.py --train True
```
test from pretrained
```
python -W ignore main.py 
```
