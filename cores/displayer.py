import logging
import time

import cv2

from utils.macros import *


def uv_to_world(uv, camera_matrix, rotation_matrix, translation_vector, deep):
    camera_coord = np.linalg.inv(camera_matrix) @ uv
    camera_coord *= deep
    world_coord = rotation_matrix @ camera_coord + translation_vector
    return world_coord


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
            deep = 12 - box_size / 1024  # TODO: deep evaluation
            deep = deep if deep > 0 else 0
            cv2.putText(frame, f'{deep}', (box[0], box[1] + int(1.2 * 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            point_center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2), tsp_frame)
            if tid in trajectories_dict:
                trajectories_dict[tid].append(point_center)
                trajectories_dict[tid] = trajectories_dict[tid][-8:]  # keep n points
            else:
                trajectories_dict[tid] = [point_center]
            trajectory = trajectories_dict[tid]
            for point_center in trajectory:
                cv2.circle(frame, point_center[:2], 2, color_id, -1)
            speeds = []
            for i in range(len(trajectory) - 1):
                point1 = np.append(np.array(trajectory[i])[:2], 1)
                point1_world = uv_to_world(point1, camera_matrix, rotation_matrix, translation_vector, deep)
                point2 = np.append(np.array(trajectory[i + 1])[:2], 1)
                point2_world = uv_to_world(point2, camera_matrix, rotation_matrix, translation_vector, deep)
                distance = np.linalg.norm(point1_world - point2_world)
                delta_t = trajectory[i + 1][2] - trajectory[i][2]
                speed = 1.0 * distance / delta_t * 3.6  # TODO: speed evaluation
                speeds.append(speed)
                # logging.info((point1_world, point2_world, distance, delta_t, speed))
            speed = np.median(speeds) if speeds else 0
            cv2.putText(frame, f'{speed} km/s', (box[0], box[1] + int(1.2 * 75)),
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
