# apt-scan

[Demo site](http://aptscan.csie.io:8000/upload/)

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

Use same parameters in [demo site](http://aptscan.csie.io:8000/upload/)

##### MSE (corner wise): top-left, top-right, down-right, down-left

##### mean MSE: mean value of MSE

##### failure rate: \#failure / \#total

|            | **TL**     | **TR**     | **DR**     | **DL**     | **mean MSE** | **failure rate** |
|------------|--------|--------|--------|--------|----------|--------------|
| **Training**   | 0.0075 | 0.0109 | 0.0122 | 0.0174 | 0.0120   | 0.0258       |
| **Validation** | 0.0105 | 0.0328 | 0.0136 | 0.0205 | 0.0194   | 0.0400       |
