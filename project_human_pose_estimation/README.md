# Project title
Human pose estimation using Deeppose

## Project report [Human pose estimation using Deeppose](https://github.com/akabiraka/cs682_computer_vision/blob/master/project_human_pose_estimation/report_presentation/Human_pose_estimation.pdf)

## Overview of directories
```
--/data
    /mpii
        /images
        /annot
--/datasets
--/models
--/output_images
--/output_models
```
1. /data contains the downloaded data. We worked only with MPII Human Pose datasets. The downloaded images should be in "images" directory.
2. /datasets contains the "dataset" implementation of pytorch so that we can provide mini-batch and other rotated and translated images to the network.
3. /models contains the implementation of the network and loss function used in the network.
4. /output_images contains the images while evaluating the model, writing the report and so on.
5. /output_models contains the "log" when we run the models.

## How to run?
We run our models in GMU Argo cluster with 64GB memory and 16GB GPU. To run in algo follow the following commands:
```
1. Go into argo
2. Copy the project directory into /scratch/your_username/
3. cd /scratch/your_username/project_path/
4. module add cuda10.1
5. sbatch job.sh
```

You can directly use the following command to run in your local machine
```
python run.py
```

To run in google-colab or in jupiter-notebook, you can use "run.ipynb". But it is not compatable with "run.py" implemented.

To run the metric evaluation and for generating other images you can use "vizualization.py". It has five distinct functions. You can run one at a time. Initially all of them will be run by folling command
```
python vizualization
```

