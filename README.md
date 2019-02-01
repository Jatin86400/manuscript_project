# Manuscript Project
**Steps to run the code**
1) Download the code and data, put data in saperate directory named 'data' such that training data lies inside /data/train/images and validation data lies inside data/train_val/images folders
2) Change the paths in train.json to your system paths.
3) Make a virtual environment of python 3.5.
4) Navigate to project directory and download all the requirements by following command,
 ```bash
 pip install -r requirements.txt
 ```
5) Now use the following command to train the model
```bash
export PYTHONPATH=$PWD
python train_new.py --exp train.json
```
6) If you get any error regarding socket connection faliure or too many files open, then increase the file opening limit to 2048 by using the command,
```bash
ulimit -n 2048
```
