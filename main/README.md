# MLAProject: Main File Overview
## Description
The "main" file in this project serves as the central hub for various components and functionalities. Here's an overview of the key elements within the file:

- Fine_Tuning Code:

Fine_Tuning script corresponds to fine-tuning the BERT model on different GLUE datasets. It encapsulates the process of training, validation, and evaluation.
Fine_Tuning_lr Code:

Fine_Tuning_lr script is designed for fine-tuning the BERT model with different learning rates, aiding in the search for optimal hyperparameters.
- Class_Fine_Tuning_V1:

Class_Fine_Tuning_V1 is a script that introduces the Fine_Tuning class. This class provides a flexible framework for initializing models, tokenizing input sentences, loading datasets, and performing training and evaluation. It supports various GLUE tasks and allows saving the model's state dictionary to a file.
- DEMO_:

The DEMO_ directory contains a demonstration showcasing the usage of different classes and methods created within the project.
- modified_transformers:

The modified_transformers directory holds a modified version of the Hugging Face Transformers library, specifically the BERT model. This version allows for architectural modifications, including the inclusion of an adapter layer within the BERT model. Details of the implementation can be found in:
main/modified_transformers/models/bert/configuration_bert.py
main/modified_transformers/models/bert/modeling_bert.py
main/modified_transformers/models/bert/BERTAdapter_V1.py

- Results:
File that contains different results wich obtains during our experimentation
# Installation
To use this project, follow these steps:

Clone the repository:

git clone [repository_url]
cd MLAProject
Navigate to the "main" directory:


cd main

Usage
Refer to the specific scripts and demos within the "main" file for detailed instructions on how to use each component of the project.
