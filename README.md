# apt-scan

### Find corners:

![hello](assets/epoch_78_7.jpg)

### Model layout:

![model](assets/network_layout.PNG)

### Warpping:

![real_life](assets/real_life.jpg)
![real_life_unwarp](assets/real_life_unwarpped.png)

![c](assets/IMG_0780.JPG)
![c_w](assets/c_unwarpped.png)

### Loss:

![loss](assets/tensorboard.PNG)

### Evaluation (lower is better):

Use same parameters in [demo site](https://github.com/peter0749/apt-scan-demo/blob/master/demo/unwrap/models.py)

##### MSE (corner wise): up-left, up-right, down-right, down-left

##### mean MSE: mean value of MSE

##### failure rate: \#failure / \#total

|            | *TL*     | *TR*     | *DR*     | *DL*     | *mean MSE* | *failure rate* |
|------------|--------|--------|--------|--------|----------|--------------|
| *Training*   | 0.0121 | 0.0237 | 0.0159 | 0.0275 | 0.0198   | 0.0405       |
| *Validation* | 0.0297 | 0.0353 | 0.0184 | 0.0567 | 0.0350   | 0.0800       |
