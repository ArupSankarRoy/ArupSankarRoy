import cv2
import os



clicked_points = []
def Points(events , x ,y , flags,param):

    frame_width, frame_height = param

    if 0 <= x < frame_width and 0 <= y < frame_height:
        if events == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x,y))
            

def video_show(cap:cv2.VideoCapture , width:int ,height:int):

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame",Points , param=(width,height))
    
    while cap.isOpened():
        ret , frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(width,height))

        for point in clicked_points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)

        cv2.imshow("frame",frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(clicked_points)

if __name__ == "__main__":
    path = os.path.join(os.getcwd(),'upload','sample_001.mp4')
    cap = cv2.VideoCapture(path)
    
    video_show(cap=cap, width=1000 , height=576)