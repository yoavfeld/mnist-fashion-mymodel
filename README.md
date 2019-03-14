# Fashion Mnist Neural Network Model
DL training model on fashion mnist DB

## Code

model.py - model training main file<br/>
callbacks.py - custom callbacks<br/>
modes.py - runnin modes (see Run section)<br/>
plot.py - ploting functions<br/>

## Run
### training:
python3.6 model.py

### calc accuracy on test set:
python3.6 model.py test

### perform single prediction:
python3.6 model.py pred

## Visualize
### plot accuracy per class:
python3.6 model.py vis

### plot confusion matrix
python3.6 model.py cm

### plot model architecture (create model.png file)
python3.6 model.py arc

## Research
experiments results can be seen in accuracy and loss plots in folder plot/ .you are more then welcome to add more.
