import boto3
import cv2
from credentials import access_key, secret_key
import os

output_dir = './data'
output_dir_img = os.path.join(output_dir, 'imgs')
output_dir_label = os.path.join(output_dir, 'anns')

# create AWS Rekognition client
reko_client = boto3.client('rekognition', region_name='us-east-1', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

# set target class
target_class = 'Zebra'

# load video
cap = cv2.VideoCapture('./zebras.mp4')

# read video frame by frame
ret = True
frame_nmr = -1
while ret:
	ret, frame = cap.read()
	if ret:
		frame_nmr += 1
		H, W, _ = frame.shape

		# convert frame to jpg
		_, buffer = cv2.imencode('.jpg', frame)

		# convert buffer to bytes
		img_bytes = buffer.tobytes()

		# detect objects
		response = reko_client.detect_labels(Image={'Bytes': img_bytes}, MinConfidence=50)
		with open(os.path.join(output_dir_label, 'frame_{}.txt'.format(str(frame_nmr).zfill(6))), 'w') as f:
			for label in response['Labels']:
				if label['Name'] == target_class:
					for instance_nmr in range(len(label['Instances'])):
						bbox = label['Instances'][instance_nmr]['BoundingBox']
						x1 = bbox['Left']
						y1 = bbox['Top']
						width = bbox['Width']
						height = bbox['Height']

						f.write('{} {} {} {} {}\n'.format(0, (x1 + width / 2) , (y1 + height / 2), width, height))

		f.close()
		cv2.imwrite(os.path.join(output_dir_img, 'frame_{}.jpg'.format(str(frame_nmr).zfill(6))), frame)
