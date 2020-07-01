# NOTE
We created a local package for installation because one of our packages (the machine learning model) was too large to upload to PyPi because of their 60 mb file size limit.

# Description
We used machine learning to identify a car's make, model, and year based solely on a photo. Our development GitHub is here: https://github.com/GaboxFH/cis4930-carfinder and our production GitHub is here: https://github.com/kbradle/CarFinder.

#### Features
- "Upload" Page
  - On desktop, you can upload an image of a car and we will tell you our prediction
  - On mobile, you can use your camera to take an image of a car and we will tell you our prediction
  - We will also display a plot with confidence percentages of our top 5 predictions
- "List of Cars" Page
  - We support the 196 cars based on the [stanford cars dataset by classes](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder)
  - We have trained for a 90% accuracy on any car on this list
  - We will be unable to identify any car not on this list

# Setup
#### Make sure that you have the following installed:
- Python 3
- WSL's Ubuntu (Or some other Linux distro)
- Optionally, use a virtual enviornment (see below)

#### Instructions
1. Install the package with `pip install dist/django-findcar-0.0.10.tar.gz`  
*THIS WILL TAKE AWHILE*

2. Run `runmyserver`

3. Go to `http://127.0.0.1:8000/` in a browser.

FindCar should be ready to use!

#### (optional) Development Instructions
To recreate the `dist/django-findcar-0.0.10.tar.gz` file, run the following command  
- `sudo python3 setup.py sdist`

# (optional) Virtual Env Setup
1. once inside the project directory run `virtualenv . -p python3`

2. run `source bin/activate` to start your env 