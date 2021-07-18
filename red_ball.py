import cv2
import numpy as np
from sort import *
from tqdm import tqdm

np.random.seed(0)

def template_demo(tpl, target, method = cv2.TM_CCORR_NORMED):
    th, tw = tpl.shape[:2]# 取高宽，不取通道 模板高宽
    global fake_color
    result = cv2.matchTemplate(target, tpl, method)
    kcf_result = np.copy(result)

    kcf_result = np.uint8(np.power((kcf_result+1)/2, 1.5)*250)
    ball_center = loc_max(kcf_result, 170)
    # print(len(ball_center))
    im_color = cv2.applyColorMap(kcf_result, cv2.COLORMAP_JET)

    match_result = []
    for i in range(len(ball_center)):
        center_y = int(ball_center[i][0] + ball_center[i][2]/2)
        center_x = int(ball_center[i][1] + ball_center[i][3]/2)
        match_result.append([ int(center_y),int(center_x),  int(center_y+th),int(center_x+tw), result[center_x, center_y]])
    cv2.imshow('kcf', im_color)
    fake_color.write(im_color)
    return match_result

    # return tl, br, lost, max_val

def loc_max(kcf_result, threshold):
    ball_center = []
    global erzhi
    ret, kcf_threshold = cv2.threshold(kcf_result, threshold, 255, cv2.THRESH_BINARY)
    #创建矩形结构单元
    g=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #形态学处理,开运算
    img_open=cv2.morphologyEx(kcf_threshold,cv2.MORPH_OPEN,g)
    _, contours, hierarchy = cv2.findContours(img_open, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        bounding_boxes = cv2.boundingRect(cnt)
        cv2.rectangle(img_open, (bounding_boxes[0], bounding_boxes[1]), (bounding_boxes[0]+bounding_boxes[2], bounding_boxes[1]+bounding_boxes[3]),(255,0,0))
        ball_center.append(bounding_boxes)
    cv2.imshow('threshold', img_open)
    img_open = np.expand_dims(img_open, 2)
    img_open = np.concatenate((img_open, img_open, img_open), axis=2)
    erzhi.write(img_open)
    return ball_center

video = cv2.VideoCapture("red_ball.flv")
bar = tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT))
# Exit if video not opened.
if not video.isOpened():
    print
    "Could not open video"
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print
    'Cannot read video file'
    sys.exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
result = cv2.VideoWriter('result.mp4',fourcc, 30.0, (frame.shape[1], frame.shape[0]))

gROI = cv2.selectROI("ROI frame", frame, False)
ROI = frame[gROI[1]:gROI[1]+gROI[3], gROI[0]:gROI[0]+gROI[2], :]
erzhi = cv2.VideoWriter('erzhihua.mp4',fourcc, 30.0, ( frame.shape[1]-gROI[2]+1, frame.shape[0]-gROI[3]+1))
fake_color = cv2.VideoWriter('jiacaise.mp4',fourcc, 30.0, ( frame.shape[1]-gROI[2]+1, frame.shape[0]-gROI[3]+1))
mot_tracker = Sort()
color = np.uint8(np.random.rand(32, 3)*255)
while True:
    ok, frame = video.read()
    bar.update(1)
    if ok:
        match_result = template_demo(ROI, frame, method=cv2.TM_CCOEFF_NORMED)
        # print(match_result)
        match_result = np.array(match_result)
        trackers = mot_tracker.update(match_result)
        for tracker in trackers:
            color_tracker = ( int(color[int(tracker[4]%30)][0]), int(color[int(tracker[4]%30)][1]), int(color[int(tracker[4]%30)][2]))
            cv2.rectangle(frame, (int(tracker[0]), int(tracker[1])), (int(tracker[2]), int(tracker[3])), color_tracker, 2)
            cv2.putText(frame, str(int(tracker[4])),(int(tracker[0]/2+tracker[2]/2), int(tracker[1]/2+tracker[3]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
        
        result.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(3) == 27:
            break
    else:
        break

result.release()
erzhi.release()
fake_color.release()
video.release()
cv2.destroyAllWindows()