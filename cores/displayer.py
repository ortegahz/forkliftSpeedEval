import logging
import time

import cv2
import numpy as np


def process_displayer(max_track_num, queue, event):
    sort_colours = np.random.rand(max_track_num, 3) * 255

    name_window = 'frame'
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)

    start_time = time.time()
    frame_count = 0

    trajectories_dict = dict()

    while True:
        tsp_frame, idx_frame, frame, online_targets = queue.get()
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
            box_size = (box[2] - box[0]) * (box[3] - box[1])
            deep = 15 - box_size / 1024  # TODO: deep evaluation
            cv2.putText(frame, f'{deep}', (box[0], box[1] + int(1.2 * 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            point_center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2), tsp_frame)
            if tid in trajectories_dict:
                trajectories_dict[tid].append(point_center)
                trajectories_dict[tid] = trajectories_dict[tid][-4:]  # keep n points
            else:
                trajectories_dict[tid] = [point_center]
            trajectory = trajectories_dict[tid]
            for point_center in trajectory:
                cv2.circle(frame, point_center[:2], 2, color_id, -1)
            speeds = []
            for i in range(len(trajectory) - 1):
                point1 = np.array(trajectory[i])[:2]
                point2 = np.array(trajectory[i + 1])[:2]
                distance = np.linalg.norm(point2 - point1)
                delta_t = trajectory[i + 1][2] - trajectory[i][2]
                speed = distance / 512 / delta_t * 3.6
                speeds.append(speed)
                # logging.info((point1, point2, distance, delta_t, speed))
            speed = np.mean(speeds) if speeds else 0
            cv2.putText(frame, f'{speed} km/s', (box[0], box[1] + int(1.2 * 100)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            info = 'tid' + ' %d' % sort_id
            cv2.putText(frame, info, (box[0], box[1] + int(1.2 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_id, 2)
        end_time = time.time()
        frame_count += 1
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f'fps: {fps:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f'fid: {idx_frame}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            event.set()
            break

    cv2.destroyAllWindows()
