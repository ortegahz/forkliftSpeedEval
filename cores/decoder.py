import logging
import time

import cv2

from utils.macros import *


def process_decoder(path_video, queue, event, buff_len=5):
    idx_frame = 0
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        logging.error('failed to open video stream !')

    t_last = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        tsp_frame = time.time()
        if not ret:
            queue.put([tsp_frame, idx_frame, None, fc])
            logging.warning('decoder exiting !')
            event.set()
            break
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        idx_frame += 1
        if queue.qsize() > buff_len:
            queue.get()
            logging.warning('dropping frame !')
        queue.put([tsp_frame, idx_frame, frame, fc])

        while time.time() - t_last < 1. / fps:
            time.sleep(0.001)
        t_last = time.time()

    cap.release()
