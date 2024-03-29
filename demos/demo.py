import argparse
import logging
import pickle
import sys
from multiprocessing import Process, Queue, Event

from cores.decoder import process_decoder
from cores.displayer import process_displayer
from utils.utils import set_logging

sys.path.append('/home/manu/nfs/ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker

sys.path.append('/home/manu/nfs/YOLOv6')
from yolov6.core.inferer import Inferer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_detector', default='/home/manu/nfs/YOLOv6/data/head.yaml', type=str)
    parser.add_argument('--weights_detector',
                        default='/home/manu/tmp/n6_ft_b64_nab_s640_dv2r_ncml_nsilu_ncont/weights/best_ckpt.pt')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--path_video',
                        default='/media/manu/data/videos/wyl.mp4')
    parser.add_argument('--path_rtsp', default='rtsp://admin:1qaz2wsx@172.20.20.58:554/h264/ch0/main/av_stream')
    parser.add_argument('--args_tracker', default='/home/manu/tmp/args_tracker.pickle')
    parser.add_argument('--max_track_num', default=100, type=int)
    parser.add_argument('--frame_rate', default=25, type=int)
    return parser.parse_args()


def run(args):
    logging.info(args)
    with open(args.args_tracker, 'rb') as f:
        args_tracker = pickle.load(f)
    logging.info(args_tracker)
    tracker = BYTETracker(args_tracker, frame_rate=args.frame_rate)
    inferer = Inferer(args.path_video, False, 0, args.weights_detector, 0, args.yaml_detector, args.img_size, False)
    stop_event = Event()

    q_decoder = Queue()
    p_decoder = Process(target=process_decoder, args=(args.path_rtsp, q_decoder, stop_event), daemon=True)
    p_decoder.start()

    q_displayer = Queue()
    p_displayer = Process(target=process_displayer, args=(args.max_track_num, q_displayer, stop_event), daemon=True)
    p_displayer.start()

    while True:
        item_frame = q_decoder.get()
        tsp_frame, idx_frame, frame, fc = item_frame
        if frame is None or stop_event.is_set():
            break
        dets = inferer.infer_custom(frame, 0.4, 0.45, None, False, 1000)
        dets = dets[:, 0:5].detach().cpu().numpy()
        online_targets = tracker.update(dets, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
        q_displayer.put([tsp_frame, idx_frame, frame, online_targets])


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
