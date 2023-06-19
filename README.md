# ETT_point
Exploring ETT detection task by first building a key point detection model

- regression.py: contains functions to
    - Pipeline that contains data loading, data augmentation, model training, model evaluation

- dataset.py: contains functions to
    - Load and preprocess data

- transforms.py: contains functions to
    - Perform data augmentation

- CXR_utils.py: contains functions to 
    - Draw bounding boxes on CXRs for visualization
    - Shrink image to desired size and shrink bounding boxes proportionally

- pipeline_utils.py: contains functions to
    - Crop images for cascaded CNN
    - Draw bounding boxes on CXRs for visualization
    - Determine if point is within bounding box