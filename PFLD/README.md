# PFLD-pytorch

Implementation of  PFLD A Practical Facial Landmark Detector by pytorch.

## Datasets

- **WFLW Dataset**

1. Download WFLW Training and Testing images
2. WFLW [Face Annotations]
3. Unzip above two packages and put them on './data/WFLW/'
4. move `Mirror98.txt` to `WFLW/WFLW_annotations`

~~~shell
$ cd data
$ python SetPreparation.py
~~~

## Training and Testing

training :
~~~shell
$ cd ..
$ python train.py
~~~

testing:
~~~shell
$ python test.py
~~~

## result
./results/example.png


