import cv2
import logging
import argparse
import warnings
import numpy as np
import csv
import os

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

import uniface

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument("--weight", type=str, default="resnet34.pt", help="Path to gaze esimation model weights")
    parser.add_argument("--view", action="store_true", default=True, help="Display the inference results")
    parser.add_argument("--source", type=str, default="assets/in_video.mp4",
                        help="Path to source video file or camera index")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name to get dataset related configs")
    parser.add_argument("--angles-out", type=str, default="",
                        help="Path to save CSV with [frame_idx,time_sec,bbox,pitch_deg,yaw_deg]. "
                             "If empty, will be auto-set to <output>.csv when --output is provided.")
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image.unsqueeze(0)


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)
    face_detector = uniface.RetinaFace()  # third-party face detection library

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occurred while loading pre-trained weights of gaze estimation model. Exception: {e}")
        raise

    gaze_detector.to(device)
    gaze_detector.eval()

    # Open source
    video_source = params.source
    if str(video_source).isdigit():
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open webcam/video source")

    # Output video (optional)
    out = None
    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, fps, (width, height))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # CSV logging
    angles_csv_path = params.angles_out
    if not angles_csv_path and params.output:
        base, _ = os.path.splitext(params.output)
        angles_csv_path = base + ".csv"

    csv_file = None
    csv_writer = None
    if angles_csv_path:
        csv_file = open(angles_csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_idx", "time_sec", "xmin", "ymin", "xmax", "ymax", "pitch_deg", "yaw_deg"])
        logging.info(f"Angles CSV: {angles_csv_path}")

    if not params.view and not params.output:
        raise Exception("At least one of --view or --output must be provided.")

    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            # Detect faces
            bboxes, keypoints = face_detector.detect(frame)
            # Iterate faces
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, frame.shape[1] - 1)
                y_max = min(y_max, frame.shape[0] - 1)
                if x_max <= x_min or y_max <= y_min:
                    continue

                # Crop & preprocess
                image = frame[y_min:y_max, x_min:x_max]
                image = pre_process(image).to(device)

                # Forward
                pitch, yaw = gaze_detector(image)
                pitch_prob, yaw_prob = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                # Bin -> degrees (for logging / display)
                pitch_deg = torch.sum(pitch_prob * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_deg = torch.sum(yaw_prob * idx_tensor, dim=1) * params.binwidth - params.angle

                # Degrees -> radians (for arrow drawing)
                pitch_rad = np.radians(pitch_deg.cpu().numpy())
                yaw_rad = np.radians(yaw_deg.cpu().numpy())

                # Draw box + gaze arrow
                draw_bbox_gaze(frame, bbox, pitch_rad, yaw_rad)

                # Overlay numeric angles (degrees)
                text = f"pitch: {float(pitch_deg.item()):.1f}°  yaw: {float(yaw_deg.item()):.1f}°"
                text_org = (x_min, max(0, y_min - 8))
                cv2.putText(frame, text, text_org, font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # Write CSV row
                if csv_writer is not None:
                    time_sec = frame_idx / float(fps)
                    csv_writer.writerow([
                        frame_idx, f"{time_sec:.3f}",
                        x_min, y_min, x_max, y_max,
                        f"{float(pitch_deg.item()):.4f}",
                        f"{float(yaw_deg.item()):.4f}",
                    ])

            # Save / show
            if out is not None:
                out.write(frame)
            if params.view:
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    if csv_file is not None:
        csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
