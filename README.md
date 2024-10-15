# neural_network_course_WS2425
This repository is the basis for the course "Neural Networks: An application-oriented introduction" at the University of Regensburg, Germany, taught by Prof. Mathias Kraus in the Winter Semester 2024/2025.

The recommended way to work with this repository is to clone it and then update the code base throughout the course.

### Colab & Kaggle
When using Colab or Kaggle, you should upload the jupyter notebook and run the following commands in a code block:

```
!git clone https://github.com/MathiasKraus/neural_network_course_WS2425
```

and

```
!pip install lightning
```

Additionally, you should change the imports as follows:

```
from data import CatDogDataModule, MNISTDataModule
```

to 

```
from neural_network_course_WS2425.data import CatDogDataModule, MNISTDataModule
```

The same needs to be done for the other Python files (model, helper).
