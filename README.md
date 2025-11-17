for webcam:
python number_plate.py

Save image using 's'
images will be saved in "D:\project python\LPR_trained\LPR\runs\detect\cam"

run ocr notebook to extract text from saved images

for any image:
python predict.py model='best.pt' source="path of image"

example:-
python predict.py model='best.pt' source="D:\project python\LPR_trained\LPR\sources\image1.jpg"


for any video:
python predict.py model='best.pt' source="D:\project python\LPR_trained\LPR\sources\demo.mp4"



best.pt model - https://drive.google.com/file/d/1IW4Hiw8C7L__o01-WTS6RMqHWpXvQeSg/view?usp=sharing
