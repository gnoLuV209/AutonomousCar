import socket
import sys
import time
import cv2
import numpy as np
import json
import base64
import torch
import math
import os
from model import build_unet

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
port = 54321

# connect to the server on local computer
s.connect(('127.0.0.1', port))


# PID
pre_t = time.time()
err_arr = np.zeros(5)


def PIDangle(err, Kp, Ki, Kd):
    global pre_t
    err_arr[1:] = err_arr[0:-1]
    err_arr[0] = err
    delta_t = time.time() - pre_t
    pre_t = time.time()
    P = Kp*err
    D = Kd*(err - err_arr[1])/delta_t
    I = Ki*np.sum(err_arr)*delta_t
    angle = P + I + D
    if abs(angle) > 25:
        angle = np.sign(angle)*25
    return int(angle)


angle = 0
speed = 70

# Load model unet
device = 'cuda:0'
checkpoint_path = "data_unet.pth"  
model = build_unet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
# Load model yolov5
model_yolo_name = './data_yolo.pt'
model_yolo = torch.hub.load('ultralytics/yolov5', 'custom',
                            path=model_yolo_name, device=0)
model_yolo.conf = 0.9

globalClass = 10
timer=0
timerSign=0
sign = 1
# main code
if __name__ == "__main__":
    try:
        """
            - Chương trình đưa cho bạn 3 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [0, 150]
            """

        while True:
            # Gửi góc lái và tốc độ để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)
            # Recive data from server
            data = s.recv(100000)
            data_recv = json.loads(data)
            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            # Yolo detect
            try:
                results_detect = model_yolo(imgage)
                # Tọa độ min
                xmin = results_detect.pandas().xyxy[0]._get_value(
                    0, 'xmin')
                ymin = results_detect.pandas().xyxy[0]._get_value(
                    0, 'ymin')
                # Tọa độ max
                xmax = results_detect.pandas().xyxy[0]._get_value(
                    0, 'xmax')
                ymax = results_detect.pandas().xyxy[0]._get_value(
                    0, 'ymax')
                labelTraffic = results_detect.pandas().xyxy[0]._get_value(
                    0, 'class')
                if labelTraffic ==0 and xmin> 280:
                    globalClass =0
                elif labelTraffic ==0 and xmin< 250:
                    globalClass =1
                elif labelTraffic == 2 or labelTraffic == 5:
                    timerSign += 1
                    if timerSign ==40:
                        globalClass = 2
                        timerSign=0
                elif labelTraffic == 3 or labelTraffic == 4:
                    timerSign += 1
                    if timerSign ==40:
                        globalClass = 3
                        timerSign=0
                elif labelTraffic == 1:
                    globalClass = 10
                    labelTraffic = 1
                sign = labelTraffic
            except:
                a=1
            print("K19_CL2A")
            # Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            imgage = cv2.imdecode(jpg_as_np, flags=1)
            # Unet detect
            img = cv2.cvtColor(imgage, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (160, 80))
            x = torch.from_numpy(img).cuda()
            x = x.transpose(1, 2).transpose(0, 1)
            x = x / 255.0
            x = x.unsqueeze(0).float()
            with torch.no_grad():
                pred = model(x)
                pred = torch.sigmoid(pred)
                pred = pred[0].squeeze()
                pred = (pred > 0.5).cpu().numpy()
                pred = np.array(pred, dtype=np.uint8)
                pred = pred * 255
            pred_resized = cv2.resize(pred, (640, 360))
            # find angle
            fixed_point = 280
            lineRow = pred_resized[fixed_point, :]
            arr = []
            for i, thresh in enumerate(lineRow):
                if thresh == 255:
                    arr.append(i)
            try:
                argmin = min(arr)
                argmax = max(arr)
                center = int((argmin+argmax)/2)
                if globalClass ==0:
                    center -=70
                elif globalClass ==1:
                    center +=70               
                angle = math.degrees(math.atan(
                    (center-pred_resized.shape[1]/2)/(pred_resized.shape[0]-fixed_point)))
            except:
                print("Detect Unstable")
            if globalClass ==0 or globalClass == 1:
                timer +=1
                if timer ==100:
                    timer=0
                    globalClass=10
                    sign =1
            if globalClass == 2 and argmax > 635:
                angle = 33
                speed = 97
                message = bytes(f"{angle} {speed}", "utf-8")
                timer +=1
                if timer ==40:
                    timer=0
                    timerSign=0
                    sign =1
                    globalClass=10
            elif globalClass == 3 and argmin <5:
                angle = -33
                speed = 97
                message = bytes(f"{angle} {speed}", "utf-8")
                timer +=1
                if timer ==40:
                    timer=0
                    timerSign=0
                    sign =1
                    globalClass=10
            elif sign != 1  and current_speed >35:
                speed = - 50
            elif abs(angle) > 35 and current_speed >60:
                speed = - 150
            else:
                speed = 140
            # Kp,Ki,Kd
            angle = PIDangle(angle, 0.4, 0.05, 0)
    finally:
        print('closing socket')
        s.close()
