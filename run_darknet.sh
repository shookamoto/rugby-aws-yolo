
#TRAIN
./darknet detector train build/darknet/x64/data/obj.data cfg/yolo-obj.cfg build/darknet/x64/darknet53.conv.74

#TEST
./darknet detector test build/darknet/x64/data/obj.data cfg/yolo-obj.cfg WEIGHT_PATH

