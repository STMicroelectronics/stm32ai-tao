# STM32AI – TAO

Welcome to STM32AI - TAO!

This repository provides a collection of example scripts that can be used to Train, Adapt and Optimize for given use cases and then easily pass the resulting AI models through STEdgeAI Dev Cloud to deploy them on your favourite STM32 boards.

The notebook scripts in [classification_tf2](./classification_tf2/) provide a complete life cycle of the model training, optimization and benchmarking using [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) and [STEdgeAI Developer Cloud](https://stm32ai.st.com/stm32-cube-ai-dc/).

While, the notebook scripts in [classification_pretrained](./classification_pretrained/) lets you download and use one of the pretrained models from pretrained [classification model repository of Nvidia TAO](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_classification/version). The models come under [creative commons license](https://creativecommons.org/licenses/by/4.0/) as mentioned on the [explainability](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_classification/explainability) link for these models. The notebook in this folder provides a demo example of how a user can download mobilenet_v2 model from the repository, convert it to onnx, and then quantize it, and finally, benchmarking using [STEdgeAI Developer Cloud](https://stm32ai.st.com/stm32-cube-ai-dc/).

NVIDIA Train Adapt Optimize (TAO) Toolkit is a simple and easy-to-use Python based AI toolkit for taking purpose-built AI models and customizing them with users' own data.

[STEdgeAI Developer Cloud](https://stm32ai-cs.st.com/home) is a free-of-charge online platform and services allowing the creation, optimization, benchmarking, and generation of AI models for the STM32 microcontrollers. It is based on the [STEdgeAI](https://www.st.com/en/development-tools/stedgeai-core.html) core technology.

Bring Your Own Model (BYOM) is a Python based package that converts any open source ONNX model to TAO compatible model. All you need is to export any model from a deep learning framework of your choice (e.g. PyTorch) to ONNX and run TAO BYOM converter.

With the Jupyter notebooks in [classification_tf2](./classification_tf2/), you will learn how to leverage the simplicity and convenience of TAO to take a pretrained model, finetune it on a sample dataset and then:
- Prune the finetuned model,
- Retrain the pruned model to recover lost accuracy,
- Export the pruned model as an onnx model,
- Quantize the model using onnxruntime,
- Run Benchmarking of the quantized onnx model (finetuned, pruned, retrained, and quantized) using STEdgeAI Developer Cloud to know the footprints and embeddability of the models.
At the end, you will have generated a trained and optimized classification model which was imported from outside TAO Toolkit, and that may be deployed via STEdgeAI Developer Cloud.

<br>

<img style="float: center;background-color: white; width: 640" src="./TAO-STM32CubeAI.png">

<br>

While with the notebook in [classification_pretrained](./classification_pretrained/) you can 
- download a pretrained model, 
- convert it to onnx and quantize it using qdq quantization with onnxruntime,
- run inference on float and quantized models, and
- Run Benchmarking of the quantized onnx model (finetuned, pruned, retrained, and quantized) using STEdgeAI Developer Cloud to know the footprints and embeddability of the models.
At the end, you will have generated a trained and optimized classification model which was imported from outside TAO Toolkit, and that may be deployed via STEdgeAI Developer Cloud.


[LICENSE](./LICENSE.md) : 

This software component is licensed by ST under BSD-3-Clause license, the "License";

You may not use this file except in compliance with the License.

You may obtain a copy of the License at: https://opensource.org/licenses/BSD-3-Clause

Copyright (c) 2023 STMicroelectronics. All rights reserved.

Copyright (c) 2023 NVIDIA. All rights reserved.

<br>

This project contains two folders:
- **[classification_tf2](./classification_tf2/)**
  - **[byom_converters](./classification_tf2/byom_converters/)** : contains Jupyter Notebooks to convert the PyTorch models to .tltb files (a proprietary template from NVIDIA for BYOM (bring your own model))
  - **[byom_person](./classification_tf2/byom_person/)** : contains Jupyter Notebooks to train the BYOM models generated using byom_converters for person-detection use case as well as the configurations for running these files in folders:
    - [specs_mobilenetv2](./classification_tf2/byom_person/specs_mobilenetv2/), and 
    - [specs_resnet18](./classification_tf2/byom_person/specs_resnet18/).
  - **[tao_person](./classification_tf2/tao_person/)** : contains a jupyter notebook `stm32ai_tao_efficientnet_b0.ipynb` to fine-tune the pretrained `efficientnet_b0` model obtained from NGC model zoo, for person-detection use case as well as the configurations for running this file in folder [specs](./classification_tf2/tao_person/specs/).
- **[classification_pretrained](./classification_pretrained/)**
  - Notebook to work with the pretrained image classification models from [classification model repository of Nvidia TAO](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_classification/version), to run the inference on them, quantize the model using onnxruntime, and then benchmark it using the [STEdgeAI Developer Cloud](https://stm32ai-cs.st.com/home).
- **[LICENSE.md](./LICENSE.md)**
- **[SECURITY.md](./SECURITY.md)**
- **[CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)**
- **[CONTRIBUTING.md](./CONTRIBUTING.md)**

## Before you start

- Create an account on myST and then sign in to [STEdgeAI Developer Cloud](https://stm32ai-cs.st.com/home) to be able access the service.
- If you don't have python already installed, you can download and install it from [here](https://www.python.org/downloads/), a **3.6 <= Python Version <= 3.8** is required to be able to use NVIDIA TAO and the scripts provided here, we recommend to use **Python v3.8.16**. (For Windows systems make sure to check the **Add python.exe to PATH** option during the installation process).
- Install Jupyter server and notebook libraries to run the Jupyter notebooks.
- Clone this repository using the following command:
```
git clone https://github.com/STMicroelectronics/stm32ai-tao.git
```
- The delivery contains three types of notebooks
    - `byom_converter_***.ipynb`,
    - `stm32ai_tao_***.ipynb`, and
    - `tao_image_classification.ipynb`.
- The users need to create seperate Python environments using conda or pyenv to run these notebooks. The Python environments can be created using following commands named as `byom_dev` (for the byom_converter_***.ipynb) and `byom_launcher` (for all the rest):
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
- activate the `byom_launcher` environment for the `stm32ai_tao_***.ipynb` and `tao_image_classification.ipynb` notebooks to adopt, optimize, benchmark and to convert your byom models in optimized c code for STM32 projects.
  - an internet connection is needed to run some cells of the notebooks
    - to download the dataset,
    - pretrained mdoels from ngc cloud,
    - to get the models from torch.hub, and
    - to connect to STEdgeAI developer cloud.