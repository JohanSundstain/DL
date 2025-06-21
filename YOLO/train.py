from ultralytics import YOLO

if __name__ == "__main__":

	model = YOLO('yolov5n.yaml') 

	# Запускаем обучение
	model.train(
		data='yolo_dataset/data.yaml', 
		epochs=50,
		imgsz=640,
		batch=1,
		name='my_yolov8_model',
		project='runs/train', 
	)
