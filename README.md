# Facial Recognition 
Implementation of basic facial recognition algorithm, using multiple face detection systems for improved accuracy.

## Specifications
### Input
Video stream, defaulting to the webcam if not specified

### Face Detection
Using OpenCV's Haar Cascade files. Currently using multiple Haar cascades for improved accuracy in detriment to performance.

### Data Gathering
* Existing pictures from the person to be recognized
* Add user function. Snaps a picture from the video stream every 3 seconds

### Model Training
OpneCV's LBPH Face Recognizer

## Environment Requirements
* python 3
* opencv
* Pillow

