#!/usr/bin/env python
# coding: utf-8
"""
Version
2022_0506_JH
GPU 선택 추가
콘솔 자동 숨김_파이참 사용시 주의
자동검사시 일시정지 기능 추가
HOME 입력시 처음영상으로 복귀
auto_test 실행시 max_inspect_time 확인가능

2022_0615
현재 image 파일명 표기
=====================================
"""
import os
import tensorflow as tf
import numpy as np
import time
import cv2
import glob
# 입력값 받은 후 콘솔 숨기기
import win32gui, win32con

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


# JH: input 값 받은 후 콘솔 숨기기
def hide_console():
    hide = win32gui.GetForegroundWindow()
    win32gui.ShowWindow(hide , win32con.SW_HIDE)


# Enable GPU dynamic memory allocation
os.environ["CUDA_VISIBLE_DEVICES"] = input('학습장비 선택 (CPU:-1, GPU0:0, GPU1:1...): ')  # jh: GPU 선택


hide_console()
Version = 'JH_2022_1028'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
    print(e)

PATH_TO_LABELS = './annotations/label_map.pbtxt'
PATH_TO_SAVED_MODEL = "./exported-models/saved_model"


# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def makedirs(path):
   try:
        os.makedirs(path)
   except OSError:
       if not os.path.isdir(path):
           raise

def load_image_into_numpy_array(path):
    img = cv2.imread(path)
    img_shape = img.shape
    return np.array(cv2.imread(path)), img_shape

# 이미지를 glob 해서 가져오는 코드
os.getcwd()
os.chdir("./images")

# jh: tset image 확장자 확인
imgNames = []
img_extension = ('*.bmp', '*.jpg')

for type in img_extension:
    imgNames.extend(glob.glob(type))

iCurImgIndex = 0
save_status = 0
auto_test = 0
max_inspect_time = 0

while(True):
    for_loop_time_start = time.perf_counter()

    image_path = imgNames[iCurImgIndex]
    image_np , img_shape = load_image_into_numpy_array(image_path)
    img_height = int(img_shape[0])
    img_width = int(img_shape[1])

    ### JH: 시간 계측 시작점
    start = time.time()
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()



    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    # 시간계측 끝(기존에 루프타임만 계산->화면표시까지)
    for_loop_time_end = time.perf_counter()
    loop_time = (for_loop_time_end - for_loop_time_start)
    strTime = "Time : %0.5f" % loop_time
    MaxTime = ", MaxTime :" + str(max_inspect_time)
    strImageIndex = 'number: ' + str(iCurImgIndex + 1)
    strImageCount = str(len(imgNames))
    cv2.putText(image_np_with_detections, strTime + MaxTime, (1, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(image_np_with_detections,  strImageIndex + ' / '+ strImageCount, (1, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
    cv2.putText(image_np_with_detections, image_path, (1, 45), cv2.FONT_HERSHEY_PLAIN, 1, (155, 155, 0))  # JH : 파일명 표기 추가
    if save_status == 1 :
        cv2.putText(image_np_with_detections, 'saving result img...ok', (1, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
    elif save_status == 2 :
        cv2.putText(image_np_with_detections, 'saving original img...ok', (1, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
    elif auto_test == 1 :
        cv2.putText(image_np_with_detections, 'auto_test...&save', (1, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

    detection_scores = sum(detections['detection_scores'] > 0.3)

    # image_size 관련부
    if (img_height > 960) | (img_width > 1280):
        # JH : 해상도가 커서 파일을 다루기 힘들때 resize
        iwidth = 1280
        iheight = 960
        adjust_width = 1280 / img_width
        adjust_height = 960 / img_height
        image_np_with_detections = cv2.resize(image_np_with_detections, dsize=(0, 0), fx=adjust_width, fy=adjust_height, interpolation=cv2.INTER_AREA)
        cv2.imshow(Version, image_np_with_detections)

    else:
        cv2.namedWindow(Version, flags=cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(Version, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(Version, image_np_with_detections)


    if auto_test == 1:
        if cv2.waitKey(100) >= 0:
            auto_test = 0
            save_status = 0
            continue
        elif loop_time > max_inspect_time:
            max_inspect_time = loop_time
        elif detection_scores >= 1:
            makedirs('../auto_test/NG')
            cv2.imwrite('../auto_test/NG/' + imgNames[iCurImgIndex], image_np)
            if iCurImgIndex == len(imgNames)-1:
                iCurImgIndex = len(imgNames)-1
                auto_test = 0
                continue
        elif detection_scores == 0:
            makedirs('../auto_test/OK')
            cv2.imwrite('../auto_test/OK/' + imgNames[iCurImgIndex], image_np)
            if iCurImgIndex == len(imgNames)-1:
                iCurImgIndex = len(imgNames)-1
                auto_test = 0
                continue
        save_status = 0
        iCurImgIndex += 1
        continue

    # JH: 키입력
    key = cv2.waitKeyEx()
    if key==0x270000:
        save_status = 0
        iCurImgIndex = iCurImgIndex + 1
        if iCurImgIndex == len(imgNames):
            iCurImgIndex = 0
        continue
    elif key==0x250000:
        save_status = 0
        iCurImgIndex = iCurImgIndex - 1
        if iCurImgIndex == -1:
            iCurImgIndex = len(imgNames) - 1
        continue
    elif key==0x240000:  # HOME
        save_status = 00
        iCurImgIndex = 0
        continue
    elif key == 0x700000: # F1
        makedirs('../save_result_img')
        cv2.imwrite('../save_result_img/' + imgNames[iCurImgIndex], image_np_with_detections)
        save_status = 1
        continue
    elif key == 0x730000: # F4
        makedirs('../save_result_img/original')
        cv2.imwrite('../save_result_img/original/' + imgNames[iCurImgIndex], image_np)
        save_status = 2
        continue
    elif key == 0x740000: # F5
        auto_test = 1
        continue
    elif (key == 0x1B) | (key == 0x2e0000):
        cv2.destroyAllWindows()
        break


