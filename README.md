# Face Recognition
Python-based face recognition, currently can recognize age and gender of faces.

## Dependencies
- Python3.5
- numpy 1.14.2
- Keras 2.0.8+
- TensorFlow 1.8.0
- opencv 3.4.1+

## Run & build manually
Tested on:
- Ubuntu 20.04 with Tensorflow 1.8.0

### install requirements
```
pip install -r requirements.txt
```

### setup environment variable
```
export DB_HOST=your_mysql_host
export DB_USER=your_db_username
export DB_PASSWORD=your_db_user_password
export DB_NAME=youd_db_name
export RUN_AT_TIME=at_what_time_will_run_everyday (example 20:00, will scan db at 8pm everydary)
```

### run the program
```
python app.py
```

## Run using Docker
Make sure docker already installed on you machine, and can work properly.

I have build docker image that contain openCV, tensorflow, pretrained model, and this code. So, just simple do:
```
docker pull matrachma/face-recognition:latest
```
That will do pulling a docker image to you local machine. Next:
```
docker run -d --restart always --name faceRecognition -v ${pwd}:/code/ -e DB_HOST=your_mysql_host -e DB_USER=your_db_username -e DB_PASSWORD=your_db_user_password -e DB_NAME=youd_db_name RUN_AT_TIME=at_what_time_will_run_everyday matrachma/face-recognition:latest python app.py
```
That command will run a docker container with name `faceRecognition` in background, and will start automatically every time you machine tuned on.
To prevent start automatically, just remove `--restart always` flag when run the container.

Stop container and remove the container, can do with:
```
docker stop faceRecognition && docker rm faceRecognition
```

## Notes
Make sure the database is exist and contain a table named `image_list`, with fields:
```
image_id int primary autoincrement
image_url varchar(2083)
image_result varchar(255)
```

When you use it for the first time, model weights are downloaded and stored in **./recognizer/pretrained_models** folder.
Or you can download it directly from:
```
https://github.com/matrachma/faceRecognition/releases/download/V1.0/weights.18-4.06.hdf5
```