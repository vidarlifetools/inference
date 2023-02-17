import sys
import gc
from numba import cuda
import argparse
import json
import logging
from os import listdir
from os.path import isfile, join
from datetime import datetime
import cv2
import os
import time
import numpy as np
import pickle
import soundfile as sf
from google.cloud import storage
import torch
from utilities.depth_compression import decode_colorized
from utilities.utils import load_video_frames, update_log, save_log


sys.path.append("")
from preprocessing.feature_detection import FeatureDetector

from dataclasses import dataclass
from preprocessing.sound_features import sound_features
import getpass
if getpass.getuser() != 'vidar':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


@dataclass
class FaceData:
    valid: bool
    face: list
    landmarks: list


@dataclass
class SkeletonData:
    valid: bool
    skeleton: list
    skeleton_2d: list

@dataclass
class SoundData:
    valid: bool
    feature: list

class UpdateFiles:
    def __init__(self, config, *args):
        print("Init Update")
        self.config = config
        self.storage_client = storage.Client.from_service_account_json(
                self.config["google_storage_key"]
            )
        self.bucket = self.storage_client.bucket("knowmeai_bucket")

    # Blob filestructure:
    # annotation/
    # annotation/<client name>/
    # annotation/<client name>/raw
    # annotation/<client name>/annotation

    def get_cloud_dict(self):
        logging.info(f"Compiling a list of files located on google storage")
        cloud_dict = {}
        #clients = []
        #raw = []
        #annotation = []
        #empty = []
        # Find client names
        blobs = self.storage_client.list_blobs(self.config["google_storage_bucket"])
        for blob in blobs:
            # Assume that the directory is listed before directory content!

            blob_name_split = blob.name.split("/")
            # If a directory only, pick the client name
            if len(blob_name_split) == 3 and blob_name_split[2] == "":
                cl = {
                    "annotation": [],
                    "raw": []
                }
                cloud_dict[blob_name_split[1]] = cl
            """
            blobs = self.storage_client.list_blobs(self.config["google_storage_bucket"])
            #bucket = self.storage_client.bucket("knowmeai_bucket")
            for blob in blobs:
            """
            # Get annotation and raw files
            if (
                len(blob_name_split) == 4
                and blob_name_split[1] in cloud_dict.keys()
                and blob_name_split[3] != ""
                and blob_name_split[3] != "inspected.json"
                and blob_name_split[3] != "xref.json"
            ):
                """
                # Check if file is empty
                blob_data = self.bucket.get_blob(blob.name)
                if blob_data != None:
                    if blob_data.size == 0:
                        empty[clients.index(split[1])].append(split[3])
                        logging.info(f"Empty file: {blob.name}")
                """
                if blob_name_split[1] == "test_client":
                    a = 1
                if blob_name_split[2] == "raw":
                    save_file = False
                    for string in self.config["file_types_to_copy"]:
                        if string in blob_name_split[3]:
                            save_file = True
                    if save_file:
                        cloud_dict[blob_name_split[1]]["raw"].append(blob_name_split[3])
                if blob_name_split[2] == "annotation":
                    cloud_dict[blob_name_split[1]]["annotation"].append(blob_name_split[3])

        # Check if all files are present and, remove from list if not
        for client in cloud_dict.keys():
            logging.info(f"Checking if all files are present and non are empty")
            delete = {}
            for raw_file in cloud_dict[client]["raw"]:
                rec_id = raw_file.split(".")[0]
                rec_id = rec_id.replace("_depth", "")
                rec_id = rec_id.replace("_imu", "")
                rec_id = rec_id.replace("_post", "")

                if rec_id + ".mp4" in cloud_dict[client]["raw"]\
                    and rec_id + ".wav" in cloud_dict[client]["raw"]\
                    and rec_id + ".json" in cloud_dict[client]["raw"]\
                    and rec_id + "_post.wav" in cloud_dict[client]["raw"]:
                    pass
                else:
                    logging.info(f"Removing incomplete file {raw_file} from {client}")
                    if client not in delete.keys():
                        delete[client] = []
                    delete[client].append(raw_file)
            """
            # Check for empty files
            logging.info(f"Remove all recorded files in case one of the files is empty")
            for i, empty_files in enumerate(empty):
                for empty_file in empty_files:
                    rec_id = empty_file.split(".")[0]
                    rec_id = rec_id.replace("_depth", "")
                    rec_id = rec_id.replace("_post", "")
                    delete[i].append(rec_id + ".json")
                    delete[i].append(rec_id + ".mp4")
                    delete[i].append(rec_id + ".wav")
                    delete[i].append(rec_id + "_post.wav")
                    delete[i].append(rec_id + "_depth.mp4")
                    delete[i].append(rec_id + ".npy.gz")
            """
        # Remove uncomplete files
        for del_client in delete.keys():
            for del_file in delete[del_client]:
                if del_file in cloud_dict[del_client]["raw"]:
                    cloud_dict[del_client]["raw"].remove(del_file)
        return cloud_dict

    def get_dest_dict(self):
        base = self.config["destination_directory"]
        clients = [f for f in listdir(base) if not isfile(join(base, f))]
        destination = []
        raw = []
        annotation = []
        dest_dict = {}
        #face = []
        #skeleton = []
        #sound = []
        for client in clients:
            dest_dict[client] = {
                "annotation": [],
                "raw": []
            }
            #raw.append([])
            #annotation.append([])
            #face.append([])
            #skeleton.append([])
            #sound.append([])
            dirs = [
                f
                for f in listdir(base + client + "/")
                if not isfile(join(base + client + "/", f))
            ]
            for dir in dirs:
                if dir == "raw":
                    files = [
                        f
                        for f in listdir(base + client + "/raw/")
                        if isfile(join(base + client + "/raw/", f))
                    ]
                    for file in files:
                        if "inspected" not in file:
                            dest_dict[client]["raw"].append(file)
                if dir == "annotation":
                    files = [
                        f
                        for f in listdir(base + client + "/annotation/")
                        if isfile(join(base + client + "/annotation/", f))
                    ]
                    for file in files:
                        if "xref" not in file:
                            dest_dict[client]["annotation"].append(file)
                """
                if dir == "face":
                    files = [f for f in listdir(base + client + "/face/")]
                    for file in files:
                        face[i].append(file)
                if dir == "skeleton":
                    files = [f for f in listdir(base + client + "/skeleton/")]
                    for file in files:
                        skeleton[i].append(file)
                if dir == "sound":
                    files = [f for f in listdir(base + client + "/sound/")]
                    for file in files:
                        sound[i].append(file)
                """
        return dest_dict

    def add_remove_annotation_files(self,client, cloud_files, dest_files):
        deleted_files = []
        added_files = []
        # Check if there are files to be deleted from the annotation directory
        for dest_client in dest_files:
            if len(dest_client["annotation"]) > 0:
                for ann in dest_client["annotation"]:
                    if ann not in cloud_files[dest_client["name"]]["annotation"]:
                        logging.info(f"Removing: {ann} from {dest_client['name']}")
                        os.remove(
                            self.config["destination_directory"]
                            + dest_client["name"]
                            + "/annotation/"
                            + ann
                        )
                        deleted_files.append("/annotation/" + ann)
            # Check if there are files to be added to the annotation directory
            logging.info(f"Checking if there are files to be added to the annotation directory")
            if len(cloud_files[dest_client["name"]]["annotation"]) > 0:
                for i, ann in enumerate(cloud_files[dest_client["name"]]["annotation"]):
                    if ann not in dest_client["annotation"]:
                        logging.info(f"Adding: {ann} to client: {dest_client['name']}")
                        cloud_file = (
                                "annotation/" + dest_client["name"] + "/annotation/" + ann
                        )
                        dest_file = (
                                self.config["destination_directory"]
                                + dest_client["name"]
                                + "/annotation/"
                                + ann
                        )
                        context_blob = self.bucket.blob(cloud_file)
                        context_blob.download_to_filename(dest_file)
                        added_files.append("/annotation/" + ann)

        return added_files, deleted_files

    def add_remove_raw_files(self, client, cloud_dict, dest_dict):
        logging.info("Create a list of active raw files referenced in the annotation files")
        deleted_files = []
        added_files = []
        for ann_file in cloud_dict[client]["annotation"]:
            json_file = (
                    self.config["destination_directory"]
                    + client
                    + "/annotation/"
                    + ann_file
            )
            raw_files = []
            with open(json_file, mode="r+") as jsonFile:
                annotation = json.load(jsonFile)
                base_file = annotation["video"].split("/")[-1].split(".")[0]
                raw_files.append(base_file + ".json")
                raw_files.append(base_file + ".mp4")
                raw_files.append(base_file + ".wav")
                raw_files.append(base_file + "_post.wav")
                raw_files.append(base_file + "_depth.mp4")
                raw_files.append(base_file + ".npy.gz")
                raw_files.append(base_file + "_imu.dat")

        # Check if there are files to be deleted from the raw directory, not referenced in a annotation file
        logging.info("Check if there are files to be deleted from the raw director")
        if len(dest_dict[client]["raw"]):
            for ann in dest_dict[client]["raw"]:
                if ann not in cloud_dict[client]["raw"] or (ann not in raw_files and not self.config["save_unannotated"]):
                    logging.info(f"Removing: {ann} from client: {client}")
                    os.remove(
                        self.config["destination_directory"]
                        + client["name"]
                        + "/raw/"
                        + ann
                    )
                    deleted_files.append("/raw/" + ann)

        # Check if there are files to be added to the raw directory
        logging.info(f"Check if there are files to be added from the raw director")
        if len(cloud_dict[client]["raw"]):
            for ann in cloud_dict[client]["raw"]:
                if ann not in dest_dict[client]["raw"] and (ann in raw_files or self.config["save_unannotated"]) \
                        and (self.config["rec_id"] == "all" or self.config["rec_id"] in ann):
                    logging.info(f"Adding: {ann} to client: {client}")
                    cloud_file = "annotation/" + client + "/raw/" + ann
                    dest_file = (
                            self.config["destination_directory"]
                            + client["name"]
                            + "/raw/"
                            + ann
                    )
                    context_blob = self.bucket.blob(cloud_file)
                    context_blob.download_to_filename(dest_file)
                    added_files.append("/raw/" + ann)
        return added_files, deleted_files

    def log_changes(self, changes):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if isfile(self.config["destination_directory"] + "labels.json"):
            os.rename(
                self.config["destination_directory"] + "labels.json",
                self.config["destination_directory"]
                + "labels-"
                + timestamp
                + ".json",
            )
        if isfile(self.config["destination_directory"] + "context.json"):
            os.rename(
                self.config["destination_directory"] + "context.json",
                self.config["destination_directory"]
                + "context-"
                + timestamp
                + ".json",
            )
        cloud_file = "annotation/labels.json"
        dest_file = self.config["destination_directory"] + "labels.json"
        context_blob = self.bucket.blob(cloud_file)
        context_blob.download_to_filename(dest_file)
        cloud_file = "annotation/context.json"
        dest_file = self.config["destination_directory"] + "context.json"
        context_blob = self.bucket.blob(cloud_file)
        context_blob.download_to_filename(dest_file)

        with open(
                self.config["destination_directory"] + "changes-" + timestamp + ".json",
                "w",
        ) as outfile:
            json.dump(changes, outfile, indent=4)

    def remove_unreferenced_raw_files(self):
        # Check that all annotation files has an existing raw file
        if self.config["save_unannotated"]:
            return
        else:
            sound_annotations = {}
            base = self.config["destination_directory"]
            clients = [f for f in listdir(base) if not isfile(join(base, f))]

            for client in clients:
                sound_annotations[client] = {}
                logging.info(f"Checking annotation files for: {client}")
                dirs = [
                    f
                    for f in listdir(base + client + "/")
                    if not isfile(join(base + client + "/", f))
                ]
                for dir in dirs:
                    if dir == "annotation":
                        files = [
                            f
                            for f in listdir(base + client + "/annotation/")
                            if isfile(join(base + client + "/annotation/", f))
                        ]
                        raw_files = [
                            f
                            for f in listdir(base + client + "/raw/")
                            if isfile(join(base + client + "/raw/", f))
                        ]
                        for raw_file in raw_files:
                            if "_post" in raw_file:
                                sound_annotations[client][raw_file.replace("_post.wav", ".wav")] = []
                        for file in files:
                            if "xref" not in file and ".json" in file:
                                json_file = join(base + client + "/annotation/", file)
                                found = False
                                with open(json_file, "r") as f:
                                    ann = json.load(f)
                                    video_file = ann["video"].split("/")[-1]
                                    if video_file not in raw_files:
                                        found = True
                                        logging.info(f"Removing annotation file : {json_file} from client: {client}")
                                    else:
                                        # Save sound annotation information for this raw file
                                        if ann["label_id"] >= 100:
                                            ann_data = {
                                                "label_id": ann["label_id"],
                                                "start": ann["start"],
                                                "end": ann["end"]
                                            }
                                            if ".mp4" in video_file:
                                                sound_annotations[client][video_file.replace(".mp4", ".wav")].append(ann_data)
                                if found:
                                    os.remove( json_file )


    def remove_files_to_be_reprocessed(self, config, dest_clients):
        if config["force_postprocess"]:
            for i, client in enumerate(dest_clients):
                if config["client"] == "all" or config["client"] == client["name"]:
                    if config["rec_id"] == "all":
                        # Remove skeleton, sound and face direcories
                        for directory_name in ["face", "skeleton", "sound"]:
                            directory = os.path.join(config["destination_directory"], client["name"], directory_name)
                            if os.path.exists(directory) and directory_name in config["postprocess"]:
                                logging.info(f"Deleting  {directory}")
                                os.system("rm -r " + directory)
                        time.sleep(1)
                    else:
                        # Remove only data directories
                        for directory_name in ["face", "skeleton", "sound"]:
                            directory = os.path.join(config["destination_directory"], client["name"], directory_name, config["rec_id"])
                            if os.path.exists(directory) and directory_name in config["postprocess"]:
                                logging.info(f"Deleting  {directory}")
                                os.system("rm -r " + directory)

    def create_feature_directories(self, config, client):
        # Check if face, skeleton and sound directories exist, create if not
        dirs = listdir(config["destination_directory"] + client["name"] + "/")
        for directory_name in ["face", "skeleton", "sound"]:
            if not directory_name in dirs:
                directory = os.path.join(config["destination_directory"], client["name"], directory_name)
                logging.info(f"Creating dir: {directory}")
                os.mkdir(directory)
        time.sleep(1)

    def process_face_and_skeleton(self, config, client, file_id, feature, log_list):
        # fetch intrinsics from the .json file
        filename = os.path.join(config["destination_directory"], client["name"], "raw", file_id + ".json")
        with open(filename, "r") as fp:
            raw_info = json.load(fp)
        if "intr_ppx" in raw_info:
            intrinsics = (raw_info["intr_ppx"], raw_info["intr_ppy"], raw_info["intr_fx"], raw_info["intr_fy"])
        else:
            # Use default setting, intrinsics from a Intel realsense D435. All new recordings shall have intrinsics
            intrinsics = (624.8223266, 361.023498, 926.580810, 925.832397)
        # Fetch the acceleration data for the recordings
        accel = None
        if "accel_x" in raw_info and config["convert_to_world"] == 1:
            if raw_info["accel_x"] != raw_info["accel_y"]:
                accel = (raw_info["accel_x"], raw_info["accel_y"], raw_info["accel_z"])

        # Generate face and skeleton frames
        if (not file_id in client["face"]) or (not file_id in client["skeleton"]):
            # Get the face image and store them frame by frame
            video_file = os.path.join(config["destination_directory"], client["name"], "raw", file_id + ".mp4")
            video_frames = load_video_frames(video_file)

            # Initialize tracking
            if "target_person" in raw_info.keys() and len(raw_info["target_person"]) > 0 and len(video_frames) > 0:
                # Initialize tracking
                target_bbox = raw_info["target_person"]
                # KCF tracker requires the first image for initialisation
                if "target_person_frame" in raw_info.keys():
                    first_frame = raw_info["target_person_frame"]
                else:
                    first_frame = 0

                if first_frame < len(video_frames):
                    feature.init_tracker(video_frames[first_frame], np.array([target_bbox]))

            face_dir = os.path.join(config["destination_directory"], client["name"], "face", file_id)
            skeleton_dir = os.path.join(config["destination_directory"], client["name"], "skeleton", file_id)
            logging.info(f"Creating: , {face_dir}, {skeleton_dir}")
            if not os.path.exists(face_dir):
                os.mkdir(face_dir)
            if not os.path.exists(skeleton_dir):
                os.mkdir(skeleton_dir)
            # Handle both color compression and depth data
            depth_file = os.path.join(config["destination_directory"], client["name"], "raw", file_id + "_depth.mp4")
            raw_depth = False
            if os.path.isfile(depth_file):
                depth_frames = load_video_frames(depth_file)
            else:
                depth_frames = []
                depth_file = os.path.join(config["destination_directory"], client["name"], "raw",
                                          file_id + ".npy")
                if not os.path.isfile(depth_file):
                    depth_file_zip = os.path.join(config["destination_directory"], client["name"], "raw",
                                                  file_id + ".npy.gz")
                    if os.path.isfile(depth_file_zip):
                        os.system("gunzip " + depth_file_zip)
                if os.path.isfile(depth_file):
                    depth_raw_frames = np.load(depth_file, allow_pickle=True)
                    # Raw depth is saved with shape: (no_of_frames * frame_height, frame_width)
                    frame_height = video_frames[0].shape[0]
                    for i in range(len(video_frames)):
                        depth_frames.append(depth_raw_frames[i * frame_height:(i + 1) * frame_height, :])
                    raw_depth = True
                else:
                    # If no depth data exsist
                    logging.info(f"No depth data found for: {video_file}")
            logging.info(f"Processing frames for: {video_file} no of frames = {len(video_frames)}")

            use_depth = True
            if len(depth_frames) == 0:
                # Assume video from mobile device, only color video
                no_of_frames = len(video_frames)
                use_depth = False
            else:
                no_of_frames = min(len(video_frames), len(depth_frames))

            # needed for filtering z-value of the face keypoints
            features = []

            for frame_no in range(no_of_frames):
                if use_depth:
                    if raw_depth:
                        depth = depth_frames[frame_no]
                    else:
                        if "min_depth" in raw_info:
                            depth = decode_colorized(depth_frames[frame_no], raw_info["min_depth"], raw_info["max_depth"],
                                                     use_disparity=True)
                        else:
                            # 1.0 to 5.0 was the values used as default before
                            depth = decode_colorized(depth_frames[frame_no], 1.0, 5.0, use_disparity=True)
                else:
                    depth = None
                if "target_person_frame" in raw_info.keys() and frame_no < raw_info["target_person_frame"]:
                    person_bbox, face_bbox, face_marks, pose2d, pose3d, faults = None, None, None, None, None, [0, ]
                else:
                    feats = feature.detect_features(video_frames[frame_no],
                                                    depth,
                                                    intrinsics=intrinsics,
                                                    accel=accel,
                                                    history=[])

                    features += [feats]
                    person_bbox, face_bbox, face_marks, pose2d, pose3d, log_data = feats
                    update_log(log_list, video_file, log_data)
                if face_bbox is not None:
                    face = video_frames[frame_no][face_bbox[1]:face_bbox[3], \
                           face_bbox[0]:face_bbox[2], :]
                    img = face
                    if face.shape[0] == 0 or face.shape[1] == 0:
                        logging.info(f"Face with 0 height or width is found {face.shape}, {face_bbox}")
                        valid_face = False
                    else:
                        valid_face = True
                else:
                    face = None
                    img = np.zeros((100, 100, 3))
                    valid_face = False

                if config["display_face"] and (img.shape[0] * img.shape[1]) > 0:
                    cv2.imshow("image", img)
                    cv2.waitKey(1)

                if "face" in config["postprocess"]:
                    with open(face_dir + "/" + str(frame_no) + ".data", "wb") as f:
                        pickle.dump(FaceData(valid_face, face, face_marks), f)

                if pose3d is not None:
                    valid_skeleton = True
                else:
                    valid_skeleton = False
                if "skeleton" in config["postprocess"]:
                    with open(skeleton_dir + "/" + str(frame_no) + ".data", "wb") as f:
                        pickle.dump(SkeletonData(valid_skeleton, pose3d, pose2d), f)
        return log_list

    def process_sound(self, config, client, file_id, sound_annotations):
        if not file_id in client["sound"]:
            sound_dir = os.path.join(config["destination_directory"], client["name"], "sound", file_id)
            os.mkdir(sound_dir)
            # Also create client, caretaker and environmental sub directories
            for dir in ["client", "caretaker", "environment"]:
                sub_dir = os.path.join(config["destination_directory"], client["name"], "sound",
                                       file_id, dir)
                os.mkdir(sub_dir)

            # Fetch configuration varibles
            #sc = json.load(open(config["sensor_configuration_file"]))["Sound"]

            sound_file = config["destination_directory"] + client["name"] + "/raw/" + file_id + ".wav"
            data, sr = sf.read(sound_file)
            if len(data.shape)>1:
                data = data[:,0]
            fps = json.load(open(config["sensor_configuration_file"]))["Camera"]["fps"]
            frame_time = 1 / fps

            start_frame = int(fps * config["sound_feature_length"])
            frame_no = start_frame
            feature_no = config["sound_feature_length"] / config["sound_feature_step"]
            fts, splits = sound_features(data, config["sound_window_length"], config["sound_window_step"], config["sound_feature_length"],
                                         config["sound_feature_step"], config["sound_sample_rate"],
                                         n_mfcc=config["sound_n_mfcc"],
                                         remove_silent=config["sound_remove_silent"], windowing=config["sound_windowing"],
                                         pre_emphasis=config["sound_pre_emphasis"], mean_normalization = config["sound_mean_norm"],
                                         noise_reduce=config["sound_noise_reduce"],
                                         use_pitch=config["use_pitch"])
            # print("Splits: ", splits)

            # Save features for the whole recording
            for idx in range(len(fts)):
                feature_no += 1
                # Save feature values for each frame
                while frame_no * frame_time < feature_no * config["sound_feature_step"]:
                    with open(sound_dir + "/" + str(frame_no) + ".data", "wb") as f:
                        pickle.dump(SoundData(fts[idx].valid, fts[idx].feature), f)
                    frame_no += 1

            # The features for the frames until start frame is set to Zero, Sound features for the first "feature_size" time is unknown
            #zero = np.zeros((len(fts[0].featureFeatureDetector)), dtype=float)
            zero = np.zeros((len(fts[0].feature)), dtype=float)
            for i in range(start_frame+1):
                with open(sound_dir + "/" + str(i) + ".data", "wb") as f:
                    pickle.dump(SoundData(False, zero), f)

            # with open(sound_dir + "/zero.data", "wb") as f:
            #    pickle.dump(SoundData(True, zero), f)

            # Save sound features for each category, remove features related to silent periods
            sub_dirs = {
                101: "client",
                102: "caretaker",
                103: "environment"
            }
            for ann in sound_annotations[client["name"]][file_id + ".wav"]:
                start_sample = int(ann["start"] * config["sound_sample_rate"])
                end_sample = int(ann["end"] * config["sound_sample_rate"])
                # Remove silent periods inside this time span
                non_silent_periods = [[start_sample, end_sample]]
                n = 0
                for split in splits:
                    if split[0] >= non_silent_periods[n][0] and split[0] <= non_silent_periods[n][1]:
                        non_silent_periods[n][0] = split[0]
                    if split[1] <= non_silent_periods[n][1] and split[1] >= non_silent_periods[n][0]:
                        non_silent_periods[n][1] = split[1]
                        non_silent_periods.append([-1, end_sample])
                        n += 1
                if non_silent_periods[n][0] == -1:
                    non_silent_periods.pop()
                # print(start_sample, end_sample, non_silent_periods)
                # Convert sample no to frame no, and copy the features category folders
                for non_silent_period in non_silent_periods:
                    non_silent_period[0] = int((non_silent_period[0] / config["sound_sample_rate"]) * fps)
                    non_silent_period[1] = int((non_silent_period[1] / config["sound_sample_rate"]) * fps)
                    for period in range(non_silent_period[0], non_silent_period[1]):
                        src = os.path.join(config["destination_directory"], client["name"], "sound",
                                           file_id, str(period) + ".data")
                        dst = os.path.join(config["destination_directory"], client["name"],
                                           "sound",
                                           file_id, sub_dirs[ann["label_id"]], str(period) + ".data")
                        # print("Copying file from ", src, " to ", dst)
                        os.system("cp " + src + " " + dst)

            """
            # Save sound features for each category
            sub_dirs = {
                101:    "client",
                102:    "caretaker",
                103:    "environment"
            }
            for ann in sound_annotations[client["name"]][file_id + ".wav"]:
                start_sample = int(ann["start"]*sc["sample_rate"])
                end_sample =int(ann["end"]*sc["sample_rate"])
                if end_sample > start_sample:
                    fts, splits = sound_features(data[start_sample:end_sample, 0],
                                                 sc["window_size"], sc["window_step"],
                                                 sc["feature_size"], sc["feature_step"],
                                                 sc["sample_rate"],
                                                 remove_silent=True, windowing=True)
                    logging.info(f"Sound anno {sub_dirs[ann['label_id']]}, {start_sample}, {end_sample}, {len(fts)}, {len(splits)}, {splits}")
                    # if more than one split (samples separated by silence), use end-sample from last
                    # split (include silence period)
                    end_sample = start_sample + splits[-1][1]
                    start_sample += splits[0][0]
                    start_frame = int((start_sample/sc["sample_rate"]) * fps)
                    end_frame = int((end_sample/sc["sample_rate"]) * fps)
                    frame_no = start_frame
                    feature_no = 0
                    first_fts = int((splits[0][0]/sc["sample_rate"])/sc["feature_step"])
                    for idx in range(first_fts, len(fts), 1):
                        feature_no += 1
                        # Save feature values for each frame
                        while (frame_no-start_frame) * frame_time < feature_no * sc["feature_step"]:
                            with open(sound_dir + "/" + sub_dirs[ann["label_id"]] + "/" + str(frame_no) + ".data", "wb") as f:
                                pickle.dump(SoundData(fts[idx].valid, fts[idx].feature), f)
                            frame_no += 1
                else:
                    logging.info(f"end sample not greater than start_sample")
            """

def main(args):
    config = json.load(open(args.config))
    update_files = UpdateFiles(config)

    #cuda_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #torch.backends.cudnn.enabled = True
    log_list = {}
    if not config["postprocess_only"]:
        # Update local files to match Google storage files
        cloud_dict = update_files.get_cloud_dict()
        dest_dict = update_files.get_dest_dict()
        changes = {}
        for client in cloud_dict.keys():
            if client == config["client"] or config["client"] == "all":
                found = [f for f in dest_dict.keys() if client == f]
                if not found:
                    # Create the directory structure for new client
                    logging.info(f"Creating directory structure for {client['name']}")
                    os.mkdir(config["destination_directory"] + client["name"])
                    os.mkdir(config["destination_directory"] + client["name"] + "/raw")
                    os.mkdir(
                        config["destination_directory"] + client["name"] + "/annotation"
                    )
                    # Add the new client to dest_clients. Since empty, all files will be copied
                    dest_dict["client"] = { "raw": [], "annotation": []}

                added_files, deleted_files = update_files.add_remove_annotation_files(client, cloud_dict, dest_dict)
                changes["annotation_files"] = {client: [{"deleted": deleted_files, "added": added_files}]}
                added_files, deleted_files = update_files.add_remove_raw_files(client, cloud_dict, dest_dict)
                changes["raw_files"] = {client: [{"deleted": deleted_files, "added": added_files }]}

        if changes["annotation_files"].keys() or changes["raw_files"].keys():
            update_files.log_changes(changes)

    update_files.remove_unreferenced_raw_files()

    # Process the videofiles and produce

    # Remove directories or files if forcing postprocessing
    dest_dict = update_files.get_dest_dict()
    remove_files_to_be_reprocessed(config, dest_clients)

    # Check if face and skeleton files are to be generated
    success = False
    current_client = None
    current_file_id = None

    while not success:

        try:
            feature = FeatureDetector(config["models_directory"],
                                      cuda_device,
                                      face_detection=config["face_detection_method"],
                                      min_pose_conf=config["min_pose_conf"],
                                      use_mediapipe=config["use_mediapipe"],
                                      mediapipe_static_mode=config["mediapipe_static_mode"])

            # Loop through all clients raw files and postprocess data according to settings
            dest_clients = get_destination_file_list(config)
            for i, client in enumerate(dest_clients):
                current_client = client["name"]
                if config["client"] == "all" or config["client"] == client["name"]:
                    logging.info(f"Postprocessing recordings for: , {client['name']}")

                    # Check if face, skeleton and sound directories exist, create if not
                    create_feature_directories(config, client)

                    # Loop through all raw files
                    if config["rec_id"] == "all":
                        raw_files = [f for f in client["raw"] if "_post.wav" in f]
                    else:
                        raw_files = [f for f in client["raw"] if "_post.wav" in f and config["rec_id"] in f]

                    for raw_file in raw_files:
                        file_id = raw_file.split("_")[0]
                        current_file_id = file_id

                        # Process face and skeleton data
                        if "face" in config["postprocess"] or "skeleton" in config["postprocess"]:
                            log_list = process_face_and_skeleton(config, client, file_id, feature, log_list)

                        if "sound" in config["postprocess"]:
                            process_sound(config, client, file_id, sound_annotations)

            success = True
        except RuntimeError as e:
            if "out of memory" not in str(e):
                # if not CUDA error, just raise it
                raise RuntimeError(e)
            else:
                # Delete the current recording from the postprocessing directories
                if current_client != None:
                    for directory_name in ["face", "skeleton", "sound"]:
                        directory = os.path.join(config["destination_directory"], current_client, directory_name, current_file_id)
                        if os.path.exists(directory):
                            logging.info(f"Deleting {directory}")
                            os.system("rm -r " + directory)

                logging.info(f"Cuda Out Of Memory error detected, restarting FeatureDetector")

                # get rid of the object
                detector = None
                del detector

                gc.collect()

                torch.cuda.empty_cache()

                # The deallocation does not occur immediately. It is added to a queue of pending deallocations. The deallocation queue is flushed automatically, but not instantly. This forces deallocations
                cuda.current_context().deallocations.clear()

                # reset feature detector
                feature = FeatureDetector(config["models_directory"],
                                          cuda_device,
                                          face_detection=config["face_detection_method"],
                                          min_pose_conf=config["min_pose_conf"])
                # wait?
                time.sleep(5)
    save_log(log_list, config["log_directory"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file", type=str, default="new_update_config.json")

    main(parser.parse_args())

