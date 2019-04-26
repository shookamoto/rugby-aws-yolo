import cv2
from darknet import *


config_path = bytes('cfg/yolo-obj.cfg', encoding = 'utf-8')
weight_path = bytes('yolo-obj_last.weights', encoding= 'utf-8')
meta_path = bytes('./build/darknet/x64/data/obj.data', encoding = 'utf-8')

#config

config_path = 'cfg/yolo-obj.cfg'
weight_path = 'yolo-obj_last.weights'
meta_path = './build/darknet/x64/data/obj.data'
detection_threshold = 0.5
video_path = 'videos'
#vid_name = '2018_Paris_7s_Match_17_Wales_vs_Spain.mp4'
vid_name = 'test_vid.mp4'
full_vid_path = video_path + "/" + vid_name
vid_out_path = 'out_video'
vid_out_name = 'out_vid.avi'
full_vid_out_path = vid_out_path + "/" + vid_out_name

#write frame to disk, then loads again... can't get c thingy working. remove at end
tmp_file = 'test.png'
#colour palette
NEON_GREEN = (22,100,8)
NEON_RED = (100,8,22)
NEON_BLUE = (8,22,100)
NEON_TURQUOISE = (8,100,85)
NEON_PURPLE = (85, 8, 100)
NEON_YELLOW = (100,85,8)
NEON_LIGHT_PURPLE = (40,8,100)
NEON_ORANGE = (100,40,8)
NEON_LIGHT_BLUE = (8,69,100)

colour_configs = {
                    'pass': NEON_RED,
                    'catch': NEON_PURPLE,
                    'line_out': NEON_LIGHT_BLUE,
                    'scrum': NEON_ORANGE,
                    'tackle': NEON_YELLOW,
                    'ruck': NEON_BLUE,
                    'try': NEON_GREEN,
                    'conversion': NEON_TURQUOISE
                    }

########


#load the video in 
cap = cv2.VideoCapture(full_vid_path)
#read one frame to get the height and width of video 
ret, frame = cap.read()
original_image_height = frame.shape[0]
original_image_width = frame.shape[1]

#copied from the darknet.py wrapper...
#make sure a copy of libdarknet.so is in the same folder... (will not run on macsm only aws at the mo..)
#file can be obtained by making the darknet projcet with LIBDARKNETSO=1

net_main = load_net_custom(config_path.encode("ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
meta_main = load_meta(meta_path.encode("ascii"))

out_vid = cv2.VideoWriter(full_vid_out_path,cv2.VideoWriter_fourcc(*'DIVX'), frameSize = (original_image_width, original_image_height), fps = 60)        
counter = 0 
while(cap.isOpened()):
    print(counter)
    counter +=1
    ret, frame = cap.read()
    #im = load_image(image_path , 0, 0)    

    #stupid... but I can't get the conversion from numpy array to cIMage working... 
    #write to disk then load again....
    cv2.imwrite(tmp_file, frame)
    im = load_image(bytes(tmp_file, encoding = 'ascii'), 0,0)

    detections = detect_image(net_main, meta_main, im, detection_threshold)
    overlay = frame.copy()
    
    #draw a bounding box for each of the detections...
    for detection in detections:
        label = detection[0].decode()
        confidence = detection[1]
        bounds = detection[2]
        #make the transparency of the box related to its confidence
        opacity = confidence
        #opencv doesn't support alpha channel!!
        col = colour_configs[label]

        #need to connect:
        # x1,y1 x2,y1
        # x2,y1 x2,y2
        # x2,y2 x1,y2
        # x1,y2 x1,y1

        y_extent = int(bounds[3])
        x_extent = int(bounds[2])
        # Coordinates are around the center
        x1 = int(bounds[0] - bounds[2]/2)
        y1 = int(bounds[1] - bounds[3]/2)
        x2 = x1 + x_extent
        y2 = y1 + y_extent
        cv2.line(overlay, (x1, y1), (x2,y1), col, thickness = 2)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        cv2.line(overlay, (x2, y1), (x2,y2), col, thickness = 2)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        cv2.line(overlay, (x2, y2), (x1,y2), col, thickness = 2)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        cv2.line(overlay, (x1, y2), (x1,y1), col, thickness = 2)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        #add the label... 
        cv2.putText(frame, "%s" % (label),(x2 + 10 , y2 + 10), cv2.FONT_HERSHEY_COMPLEX, 1.0, col, 3)
        
    out_vid.write(frame)
    if not ret:
        print("finished processing")
        out_vid.release()
        cv2.destroyAllWindows()
        #convert using ffmpeg
        #subprocess.call('ffmpeg -y -i %s %s' % (vid_name, vid_name_mp4), shell= True)

