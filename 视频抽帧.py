import cv2
import os


def get_source(rootpath:str,source_format:list=["mp4"]):
    source_list = [i for i in os.listdir(rootpath) if i.split(".")[1] in source_format]
    return [os.path.join(rootpath,i) for i in source_list]

def get_images(videopath,destination,step = 5):
    if not os.path.exists(destination):
        os.makedirs(destination)

    video_name = os.path.splitext(os.path.split(videopath)[1])[0]
    cap = cv2.VideoCapture(videopath)
    # print(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    count = 1
    while ret:
        if count % step == 0 and ret:
            filename = os.path.join(destination, video_name + "_" + str(count) + ".jpg")
            cv2.imwrite(filename, frame)
        ret, frame = cap.read()
        count += 1
    cap.release()
    cv2.destroyAllWindows()


destination = "./images/"
for videopath in get_source("./videos/output"):
    get_images(videopath,destination,step=15)

# get_images("./main_code/videos/positive1.mp4",destination="./main_code/images/",step=10)
