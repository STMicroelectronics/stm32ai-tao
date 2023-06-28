# STM32AI – TAO

Welcome to STM32AI - TAO!

This repository provides a collection of example scripts that can be used to Train, Adapt and Optimize for given use cases and then easily pass the resulting AI models through STM32Cube.AI to deploy them on your STM32 boards.

These scripts provide a complete life cycle of the model training, optimization and benchmarking using Nvidia TAO and STM32Cube.AI Developer Cloud.

This project contains two folders:
- **[byom_converters](./classification_tf2/byom_converters/)** : contains Jupyter Notebooks to convert the PyTorch models to .tltb files (a proprietary template from Nvidia for BYOM (bring your own model))
- **[byom_person](./classification_tf2/byom_person/)** : contains Jupyter Notebooks to train the BYOM models generated using byom_converters for person-detection use case as well as the configurations for running these files in folders:
   - [specs_mobilenetv2](./classification_tf2/byom_person/specs_mobilenetv2/), and 
   - [specs_resnet18](./classification_tf2/byom_person/specs_resnet18/).
- **[LICENSE.md](./LICENSE.md)**
- **[SECURITY.md](./SECURITY.md)**
- **[CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)**
- **[CONTRIBUTING.md](./CONTRIBUTING.md)**


## Before you start

- Create an account on myST and then sign in to [STM32Cube.AI Developer Cloud](https://stm32ai-cs.st.com/home) to be able access the service.
- If you don't have python already installed, you can download and install it from [here](https://www.python.org/downloads/), a **3.6 <= Python Version <= 3.8** is required to be able to use Nvidia TAO and the scripts provided here, we recommend to use **Python v3.8.16**. (For Windows systems make sure to check the **Add python.exe to PATH** option during the installation process).
- Install Jupyter server and notebook libraries to run the Jupyter notebooks.
- Clone this repository using the following command:
```
git clone https://github.com/STMicroelectronics/stm32ai-tao.git
```
- The delivery contains two types of notebooks
    - `byom_converter_***.ipynb`, and
    - `stm32ai_tao_***.ipynb`.
- The users need to create seperate Python environments using conda or pyenv to run these notebooks. The Python environments can be created using following commands named as `byom_dev` and `byom_launcher`:
```
cd stm32ai-tao
python -m venv <env-name>
```
- Activate your virtual environment, on Windows run:
 ```
<env-name>\Scripts\activate.bat
```
On Unix or MacOS, run:
 ```
source <env-name>/bin/activate
```

**NOTE**: The names of the environments are just a suggestion and users can choose whichever name they prefer.

## Running the Jupyter Notebooks
The running of Jupyter notebooks requires:
- activate the `byom_dev` environment for all the `byom_converter***.ipynb` notebooks to convert your onnx models to byom model (.tltb), and
- activate the `byom_launcher` environment for the `stm32ai_tao_***.ipynb` notebooks to train, adopt, optimize, benchmark and to convert your byom models in optimized c code for STM32 projects.
  - an internet connection is needed to run some cells of the notebooks
    - to download the dataset,
    - to get the models from torch.hub, and
    - to connect to STM32Cube.AI developer cloud.
