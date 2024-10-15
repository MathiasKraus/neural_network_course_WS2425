# neural_network_course_WS2425

### Colab & Kaggle
When using Colab or Kaggle, you should upload the jupyter notebook and run the following commands in a code block:

```
!git clone https://github.com/MathiasKraus/neural_network_course_WS2425
```

and

```
!pip install lightning
```

### Change imports
In case of import errors, change 

```
from data import CatDogDataModule, MNISTDataModule
```

to 

```
from neural_network_course_WS2425.data import CatDogDataModule, MNISTDataModule
```

Do the same for the other Python files (model, helper)
