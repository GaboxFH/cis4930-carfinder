from imutils import paths
import argparse
import requests
import cv2
import os

# Code Citation #########################
#    Title: download_images.py
#    Author: Adrian Rosebrock
#    Date: Originally posted December 4, 2017
#    Code version: N/A
#    Availability: https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
# #######################################

# construct the argument parse and parse the arguments given from command line
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True, help="path to .txt file containing list of image URLs")
ap.add_argument("-o", "--output", required=True, help="path to output directory for downloaded images")
args = vars(ap.parse_args())
# grab the list of URLs from the input file, then initialize the images downloaded count to 0
rows = open(args["urls"]).read().strip().split("\n")
total = 0
# loop the URLs from the .txt file
for url in rows:
	try:
		# try to download the image
		r = requests.get(url, timeout=60)
		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(8))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()
		print("[INFO] downloaded: {}".format(p))
        # increment total count
		total += 1
	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading {}...skipping".format(p))
# loop over the image paths we just downloaded
for imagePath in paths.list_images(args["output"]):
	# initialize as false
	delete = False
	# try to load the image
	try:
		image = cv2.imread(imagePath)
		# if the image is equal to `None` then it could not be properly loaded (probably corrupted) so delete it
		if image is None:
			delete = True

	except:
        # if exception encountered when you try to load image then the image will be unusable for dataset, delete it
		print("Except")
		delete = True
	# encountered if image is corrupted or could not be loaded
	if delete:
		print("[INFO] deleting {}".format(imagePath))
		os.remove(imagePath)
