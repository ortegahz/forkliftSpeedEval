import cv2
import time
import numpy as np


def process_displayer(max_track_num, queue):
    sort_colours = np.random.rand(max_track_num, 3) * 255

    name_window = 'frame'
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)

    start_time = time.time()
    frame_count = 0

    while True:
        idx_frame, frame, online_targets = queue.get()
        for track in online_targets:
            tlwh = track.tlwh
            tid = track.track_id
            tlbr = [tlwh[0], tlwh[1], tlwh[2] + tlwh[0], tlwh[3] + tlwh[1]]
            bbox = np.concatenate((tlbr, [tid + 1])).reshape(1, -1)
            bbox = np.squeeze(bbox)
            box = bbox.astype(int)
            sort_id = box[4].astype(np.int32)
            color_id = sort_colours[sort_id % max_track_num, :]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_id, 2)
            info = 'tid' + ' %d' % sort_id
            fontScale = 1.2
            cv2.putText(frame, info,
                        (box[0], box[1] + int(fontScale * 25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id,
                        2)
        end_time = time.time()
        frame_count += 1
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'{idx_frame}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 2)
        cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
