#比如使用yolov5加追踪，其中追踪部分使用boxmot封装的比较方便，参见https://github.com/mikel-brostrom/boxmot
import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT


tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=False,
)

vid = cv2.VideoCapture(0)

while True:
    ret, im = vid.read()

    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])

    # Check if there are any detections
    if dets.size > 0:
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
    # If no detections, make prediction ahead
    else:
        dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
    tracker.plot_results(im, show_trajectories=True)

    # break on pressing q or space
    cv2.imshow('BoxMOT detection', im)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()