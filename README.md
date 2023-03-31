# AlphaGo-Zero-Implementation-Go



### Train a 9*9 Game model 

To start training a model for Go, run three python script concurentlly:

To generate game examples. (can run multiple times of this script in a concurent manner)
```bash
python runSelfplay.py
```

To generate network weight using examples.
```bash
python runTraining4tars.py
```
For training 9*9 model, run this line 4 times and do not forget to modify the name of network's tar file each time in the Coach.

When 4 network weight is ready, update the best network weight with the best of these 4.
```bash
python runTrainingBest.py
```
### Play with model
```bash
python pit.py
```
