# Description
We used machine learning to identify a car's make, model, and year based solely on a photo.

#### Features
- "Upload" Page
  - On desktop, you can upload an image of a car and we will tell you our prediction
  - On mobile, you can use your camera to take an image of a car and we will tell you our prediction
  - We will also display a plot with confidence percentages of our top 5 predictions
- "List of Cars" Page
  - We support the 196 cars based on the [stanford cars dataset by classes](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder)
  - We also support an additional 30 cars (from 2017 - 2019) web-scraped from Gooogle:
    - Cheverlot Impala
    - Chevrolet Silverado
    - Ford Explorer
    - Ford Focus
    - Honda Accord
    - Honda Civic
    - Nissan Altima
    - Nissan Maxima
    - Toyota Camry
    - Toyota Corolla
  - We have trained for a 74% accuracy on any car on this list
  - We are unable to identify any car not on this list
  - We are unable to tell whether an image is a car or not

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

# (optional) Development Setup
#### Build
1. To recreate the `dist/django-findcar-0.0.10.tar.gz` file, run the following command  
  - `sudo python3 setup.py sdist`
2. Then run the Setup Instructions above

#### Run Source Code Directly
1. `pip install -r requirements.txt`

2. `python3 manage.py runserver`

3. Go to `http://127.0.0.1:8000/` in a browser.

# (optional) Virtual Env Setup
1. once inside the project directory run `virtualenv . -p python3`

2. run `source bin/activate` to start your env 
