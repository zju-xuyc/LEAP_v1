## DATASET PREPARATION
### VIDEO PREPARATION
We divided video into **Train**,**Val**,**Test** three parts.

We recommand that train.mp4 and val.mp4 should be last for at least 1 hours to get better performance.

Test.mp4 is for the real test and can last for customized time.

### Config preparation
Please add **dataset_name.yaml** into ```./configs``` following the example file ```adventure.yaml```

Detector configs is in ```./configs/defaults.py```

### PREPROCESSING
#### Train ReID Model
```cd fast_reid_train```

Please follow the instrcutions in ```run.sh```

Generate the reid datasets by running 

```python
python ./tools/generate_reid.py
```

#### Clustering dataset
```bash
python main.py --load 
```

## Install Dependency
### One bug fix
Find
```python
/opt/homebrew/lib/pythonX.X/site-packages/torch/nn/modules/upsampling.py
```
in line 153-154:


Change 
```python
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,recompute_scale_factor=self.recompute_scale_factor)
```

To 
```python
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
# recompute_scale_factor=self.recompute_scale_factor)
```


### OPTIONAL FUNCTIONS
#### Using cache to optimzing I/O
Convert videos into frames seperately and store on SSD disc to get better I/O performance.
```python
python tools.data_prepare.py
```
Use function **dump_pic()**