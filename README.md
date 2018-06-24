# apt-scan

### Training:

![hello](assets/epoch_78_7.jpg)

### Wrapping:

![real_life](assets/real_life.jpg)
![real_life_unwarp](assets/real_life_unwarpped.png)

![c](assets/IMG_0780.JPG)
![c_w](assets/c_unwarpped.png)

### Loss:

![loss](assets/tensorboard.PNG)

### Evaluation (lower is better):

Use same parameters in [demo site](https://github.com/peter0749/apt-scan-demo/blob/master/demo/unwrap/models.py)

##### SSE (corner wise): up-left, up-right, down-right, down-left

##### mean SSE: mean value of SSE

##### failure rate: \#failure / \#total

![evaluation_train](assets/sse_train.PNG)

(training)

![evaluation](assets/sse.PNG)

(validation)