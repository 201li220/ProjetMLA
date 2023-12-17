# Results of the trainings performed on the GPU

The three folder Run_Adapter_64, Run_256 and Run_Fine_Tuning contains the main results used for in the paper, with a notebook to visualize the results and a config.json file representing the model that can be viewed using the view_config function.

The folders TopLayers and TopNoEmb respectively represent the fine-tuning with and without Word Embedding being trained, this will be used to estimate if classic methods can also be parameter-efficient.

Others contains all other trainings made during the project.
