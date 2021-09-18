# Tensorflow Facial Recognition

A facial recognition software created using Tensorflow 2.6

## Table of Contents

1. [Introduction](#intro)
2. [Setup](#setup)
    - [2.1. Python Environment](#pyenv)
        - [2.1.1. Install virtualenv](#insvenv)
        - [2.1.2. Create environment](#crenvpy)
        - [2.1.3. Activate environment](#actenvpy)
        - [2.1.4. Installing dependencies](#insdeps)
        - [2.1.5. Deactivate environment](#deactenvpy)
    - [2.2. Conda Envrionment](#conenv)
        - [2.2.1. Create environment](#crenvcon)
        - [2.2.2. Activate environment](#actenvcon)
        - [2.2.3. Deactivate environment](#deactenvcon)
3. Configurations
    - 3.1. Architecture
    - 3.2. Train
    - 3.3. Test
4. Model Architecture
5. Train
    - 5.1. Dataset Preparation
    - 5.2. Syntax
    - 5.3. Arguments
    - 5.4. Classes
    - 5.5. Checkpoints
    - 5.6. Models
    - 5.7. Tensorboard
6. Evaluate
    - 6.1. Dataset Preparation
    - 6.2. Syntax
    - 6.3. Arguments
7. Predictions[WIP]
8. TFRecords[WIP]
    - 8.1. Create TFRecords
    - 8.2. Train
    - 8.3. Evaluate
9. Training Attempts
10. Models - trained model example using faces dataset and link to the dataset
11. Dependencies

## 1. Introduction <a name="intro"></a>

- This is a project created for facial recognition. 

- This project contains of a [Convolutional Neural Network(CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) model and it can be  trained for a dataset containing faces of the people that need to be detected/recognized by the model.

- After a model is trained for a dataset, it can be used to recognize faces that were part of the training dataset. 

## 2. Setup <a name="setup"></a>

- In order to run this script, an environment needs to be created and all the dependencies of this code should be installed to the created environment.

- The script was developed using **Python 3.8.8** and thus the recommended version for running scripts in this repository.

- There are two options for this to be done. You can either use the Python Environment option or the Conda Environment option.

- **Setting up a Python environment is much easier than setting up a Conda environment**. If you've never worked with either of them, I suggest the first option. If you are familiar with Conda and have Conda installed on your computer, the Conda environment option is for you.

### 2.1. Python Environment <a name="pyenv"></a>

#### 2.1.1. Install virtualenv <a name="insvenv"></a>

- A Python virtual environment is created using the Python package, `virtualenv`.

- `virtualenv` can be installed using the follwing command.

    ```
    pip install virtualenv
    ```
#### 2.1.2. Create environment <a name="crenvpy"></a>

- After installing `virtualenv`, it can be used to create a python virtual environment.

- Use the following script to create a virtual enviornment named *facial_recog* in the current directory you're on.

    ```
    virtualenv facial_recog
    ```

#### 2.1.3. Activate environment <a name="actenvpy"></a>

- The process of activating a virtual environment varies with the Operating System and below are separate guides to activate a virtual environment on Linux and Windows.

    ##### Linux

- Use the following command to activate the created virtual environment on Linux.

    ```
    source facial_recog/bin/activate
    ```

    ##### Windows

- Use the following command to activate the created virtual environment on Windows.

    ```
    facial_recog/Scripts/activate
    ```

#### 2.1.4. Installing dependencies <a name="insdeps"></a>

- Install required dependencies to the activated virtual environment using the following command. **Make sure you are executing the command in the main directory of the repository, where `requirements.txt` is located**.

    ```
    pip install -r requirements.txt
    ```
**After installing dependencies, you can run the scripts on the virtual environment created.**

#### 2.1.5. Deactivate environment <a name="deactenvpy"></a>

- The following command will deactivate the virtual environment.

    ```
    deactivate
    ```

### 2.2. Conda Environment <a name="conenv"></a>

Make sure you have successfully installed [Conda](https://docs.conda.io/en/latest/) on your computer.

#### 2.2.1. Create environment <a name="crenvcon"></a>

- Use the following command to create the conda environment with requirements already installed. **Make sure you are executing the command in the main directory of the repository, where `env.yml` is located**

    ```
    conda env create -f env.yml
    ```

#### 2.2.2. Activate environment <a name="actenvcon"></a>

- Use the following command to activate the created environment.

    ```
    conda activate facial_recog
    ```

#### 2.2.3. Deactivate environment <a name="deactenvcon"></a>

- Use the following command to deactivate the Conda environment.

    ```
    conda deactivate
    ```

If you followed the above steps properly, now you're ready to run scripts.

## 3. Configurations

- All configurations can be found at `config.py`. 

- Configurations are divided into *Architecture*, *Train* and *Test* for ease of use.

### 3.1. Architecture

| Config | Description | Notes |
|--------|-------------|-------|
| `image_height` | Height of an image processed by scripts | Will be used in reading images and setting model input shape |
| `image_width` | Width of an image processed by scripts | Will be used in reading images and setting model input shape |
| `input_channels` | Number of channels in input images | Use default **3** for RGB images. |
| `hidden_layers` | Number of hidden layers(Dense layers) in the model | Use less layers for simple models and more layers for more complicated models |
| `dropout_rate` | Dropuot rate to be used in Dropout layers in the model | Use a value between 0 and 1 |

### 3.2. Train

| Config | Description | Notes |
|--------|-------------|-------|
| `batch_size` | Size of image batches used in loading images, training  | Suggested values: 8, 16, 32, 64, 128 |
| `epochs` | Number of epochs the model should trin for | An integer value must be used |
| `learning_rate` | Learning rate of the optimixer algorithm | Can be changed to prevent overfitting/underfitting |
| `patience_epochs` | Number of epochs with no improvement after which training will be stopped | If the validation accuracy does not improve for this many epochs, training process will stoped |
| `validation_split` | Portion of the training dataset that is plit for validation | Use a value between 0 and 1 |

### 3.3. Test

| Config | Description | Notes |
|--------|-------------|-------|
| `batch_size` | Size of image batches used in evaluating models | Suggested values: 8, 16, 32, 64, 128 | 

