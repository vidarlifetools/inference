import numpy as np
import pickle
import vg
from constants import *
from utilities.compound_recognition import do_compound_recognition,\
    predict_pose_class, compute_feature_vector_gesture, initialize_buffer_info,\
    get_sequence_from_buffer_info, update_buffer_info
class compound_prediction:
    def __init__(self, model_filename_pose, model_filename_gesture, logger):
        with open(model_filename_pose, "rb") as file:
            self.pose_model = pickle.load(file)
        with open(model_filename_gesture, "rb") as file:
            self.gesture_model = pickle.load(file)
        self.nof_pose_classes = len(self.pose_model.classes_)
        self.logger = logger
        self.buffer_info = initialize_buffer_info()

    def compound_class(self, keypoints):

        # Load person dict and labels dict
        with open(os.path.join(base_for_person, person, ML_CAT, LABELLED_PERSON_FILENAME), 'r') as infile:
            person_dict = json.load(infile)
        label_json_filename = person_dict['labels_file']
        with open(label_json_filename, 'r') as infile:
            labels_dict = json.load(infile)

        # Get nr of labels for each of the recognition modules -> face, gesture and sound
        nof_expr_classes, nof_gesture_classes, nof_sound_classes = get_nof_submodel_labels(labels_dict)

        # Load compound model
        compound_model = load_compound_module(person_dict['model_filename_compound'])

        # Initialize empty arrays
        expression_frame_nos = []
        expression_classes = []
        gesture_frame_sequences = []
        gesture_classes = []
        sound_frame_nos = []
        sound_classes = []

        # Initialize time_stamps
        if not INCLUDE_SOUND:
            timestamp_sound = [-1]
        if not INCLUDE_FACE_EXPR:
            timestamp_face = [-1]
        if not INCLUDE_GESTURE:
            timestamp_gesture = [-1]

        # TODO Offset, timestamp_count should be constants?
        while True:
            # Filling in timestamp and class information related to face expression
            if INCLUDE_FACE_EXPR:
                timestamp_face, _, class_face_nos, class_face_probs = get_class_stream(sub_face_socket,
                                                                                       offset=10,
                                                                                       timestamp_count=1)
                expression_frame_nos += [(int(np.round(timestamp_face[0] * frame_rate)))]
                expression_classes += [class_face_nos[0]]

            # Filling in timestamp and class information related to gesture
            if INCLUDE_GESTURE:
                timestamp_seq_gesture, _, class_gesture_nos, class_gesture_probs = get_class_stream(sub_gesture_socket,
                                                                                                    offset=8,
                                                                                                    timestamp_count=2)
                timestamp_gesture = [timestamp_seq_gesture[1]]
                gesture_frame_sequences += [[int(np.round(t * frame_rate)) for t in timestamp_seq_gesture]]
                gesture_classes += [class_gesture_nos[0]]

            # Filling in timestamp and class information related to sound expression
            if INCLUDE_SOUND:
                timestamp_sound, _, class_sound_nos, class_sound_probs = get_class_stream(sub_sound_socket,
                                                                                          offset=6,
                                                                                          timestamp_count=1)
                sound_frame_nos += [(int(np.round(timestamp_sound[0] * frame_rate)))]
                sound_classes += [class_sound_nos[0]]

            # Get frame number
            if INCLUDE_GESTURE:
                frame_no = int(np.round(timestamp_gesture[0] * frame_rate))
            elif INCLUDE_FACE_EXPR:
                frame_no = int(np.round(timestamp_face[0] * frame_rate))
            elif INCLUDE_SOUND:
                frame_no = int(np.round(timestamp_sound[0] * frame_rate))

            # Do compound recognition if buffer is long enough
            if do_compound_recognition(frame_no):
                # Check that timestamp face and timestamp gesture are the same, otherwise AssertionError is raised
                if INCLUDE_FACE_EXPR and INCLUDE_GESTURE:
                    assert abs(timestamp_face[0] - timestamp_gesture[0]) < 0.00001

                # Get window feature vector compound, current frame nr is the last frame in the window
                first_frame = frame_no - COMPOUND_BUFFER_LEN + 1
                last_frame = frame_no + 1
                frame_sequence = (first_frame, last_frame)

                # Slice the arrays for each recognition module
                sequence_expr_classes, expr_frame_nos = get_sequence_labels(frame_sequence, expression_classes)
                sequence_sound_classes, sound_frame_nos = get_sequence_labels(frame_sequence, sound_classes)
                sequence_gesture_classes, _ = get_sequence_labels(frame_sequence, gesture_classes)
                gesture_sequence = gesture_frame_sequences[first_frame:last_frame]

                # Calculate compound feature vector from the sequence
                fts, fts_valid = compute_feature_vector_compound(
                    expr_frame_nos, sequence_expr_classes, nof_expr_classes,
                    gesture_sequence, sequence_gesture_classes, nof_gesture_classes,
                    sound_frame_nos, sequence_sound_classes, nof_sound_classes,
                    frame_sequence, logger, frame_no)

                # Reshape array for model prediction
                fts = np.array(fts).reshape(1, -1)

                # Compound model prediction, log the results
                if compound_model is None or not fts_valid:
                    class_nos = np.array([MISSING_CLASS], dtype=np.uint8)
                    class_scores = np.array([0], dtype=np.float16)
                else:
                    class_nos = np.array(compound_model.predict(fts), dtype=np.uint8)
                    class_scores = np.array(compound_model.decision_function(fts), dtype=np.float16).ravel()
                log_compound_fts(logger, frame_no, fts, class_nos, class_scores)

                # Send compound stream
                timestamp = np.array([timestamp_gesture[0]], dtype=np.float64)
                send_compound_stream(pub_socket, timestamp, class_nos)

                # Write fts to file for testing
                if WRITE_FEATURES_LABEL_TRAIN_TEST:
                    write_feature_vector(np.concatenate((np.array([int(frame_no) + 1]).reshape((1, 1)), fts), axis=1),
                                         class_nos[0], 'fts_compound_test', 'cl_compound_test')

            # If end of video is reached, predict class as missing and stop this thread
            if end_of_video(timestamp_face, timestamp_gesture, timestamp_sound):
                class_nos = np.array([MISSING_CLASS], dtype=np.uint8)
                timestamp = np.array([-1], dtype=np.float64)
                send_compound_stream(pub_socket, timestamp, class_nos)
                break

        # End of thread

        return class_nos


