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
3. Train
    - 3.1. Dataset Preparation
    - 3.2. Classes
    - 3.3. Checkpoints
    - 3.4. Models
    - 3.5. Tensorboard
4. Evaluate
5. Predictions[WIP]
6. TFRecords[WIP]
    - 6.1. Create TFRecords
    - 6.2. Train
    - 6.3. Evaluate
7. Training Attempts
8. Models - trained model example using faces dataset and link to the dataset
9. Dependencies

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

If you folled the above steps properly, now you're ready to run scripts.
