# QuaLITi: QML Workload Optimization
Code implementation of the paper *"<ins>Qua</ins>LITi: Quantum Machine <ins>L</ins>earning Hardware Selection for <ins>I</ins>nferencing with Top-<ins>Ti</ins>er Performance"*. In this work, we propose a novel multi hardware training setup for training QML workloads to optimize hardware wait time.

## Important files:
This repository contains 3 main files pertaining to three main steps as shown in Fig. 2 of the paper.

1. <b>Model selection:</b> ```model_selection.py``` contains this step. The user can train the model for all the specified ranges (1-4) and choose the best model with highest inferencing values. While not explicitly coded, the user can manually modify this code to switch from training to inferencing to perform the inferencing runs.

2. <b>Multi-hardware training:</b> ```multi_hardware_training.py``` does this task. The best range values chosen for each dataset are ```r = 1``` for ```iris``` and ```digits89``` datasets and ```r = 3``` for ```digits01``` dataset. The user can specify the desired dataset and configuration and train for 2 epochs. Number of epochs can be changed if needed.

3. <b>Inferencing trained model:</b> ```multi_hardware_inferencing.py``` performs this task. Once again, the user can select the desired dataset and configuration for inferencing. As mentioned in the paper, the final inferencing accuracy is averaged for 10 inferencing runs. 

## Extra files
We also provide some additional code files that perform more analysis from the paper

1. ```motivation_example.py``` contains code that trains the motivational example from Fig. 1

2. ```queue_depth.py``` is used to compute real time queue depth of a real IBM quantum hardware

3. ```cifar_noisy_training.py``` and ```cifar_inferencing.py``` are used to train and inference binary classes of Cifar-10 dataset on a hybrid classical-quantum neural network using the proposed multi-hardware training setup. 

4. ```pennylane_transpilation_test.ipynb``` is an interactive ipython file used to analyze layerwise depth of a single Strongly Entangling Layer post-transpilation. The user can modify range value and see how the depth changes.

5. ```tsne_cifar.ipynb``` uses code to figure out classifiable classes by plotting tsne prior to classification using QNN. This is more of a trial-and-error process and it is found out that classes 0 and 6 have reasonable separation for their tsne plots.

6. ```utils.py``` majorly contains code to obtain a subset backend consisting of 8 qubits out of a larger hardware containing 20/27/127 qubits. This is done to significantly cut down simulation time as using all qubits at once in simulation yields infeasible compute time.

## Trained models
All the trained models are present in ```trained_models``` folder. We have added trained models for motivational example, model selection, multi-hardware training and cifar-10 binary classification. The users can validate our results using these models. One thing to note is that due to presence of noise, the users may observe some fluctuation in the results which numerically may not yield exactly the same values. However, the trend observed will be the exact same. 

## Python Library versions
The following are all the relevant library versions for all python libraries used:

```
python==3.10.12
torch==2.2.0
torchvision==0.17.0
pennylane==0.34.0
qiskit==1.0.0
scikit-learn==1.4.0
numpy==1.26.4
tqdm==4.66.2
ipython==8.21.0
matplotlib==3.8.3
qiskit-machine-learning==0.7.1
qiskit-aer==0.13.3
qiskit-algorithms==0.2.2
```

These libraries are not necessarily required to be at their respective above mentioned versions, however the code has been tried and tested successfully to work at these versions.