import numpy as np
import pickle
import vg

mediapipe_landmarks = [
    61,     # 00 Mouth end (right)
    292,    # 01 Mouth end (left)
	0,      # 02 Upper lip (middle)
	17,	    # 03 Lower lip (middle)
	50,	    # 04 Right cheek
	280,	# 05 Left cheek
	48,	    # 06 Nose right end
	4,	    # 07 Nose tip
	289,	# 08 Nose left end
	206,	# 09 Upper jaw (right)
	426,	# 10 Upper jaw (left)
	133,	# 11 Right eye (inner)
	130,	# 12 Right eye (outer)
	159,	# 13 Right upper eyelid (middle)
	145,	# 14 Right lower eyelid (middle)
	362,	# 15 Left eye (inner)
	359,	# 16 Left eye (outer)
	386,	# 17 Left upper eyelid (middle)
	374,	# 18 Left lower eyelid (middle)
	122,	# 19 Nose bridge (right)
	351,	# 20 Nose bridge (left)
	46,	    # 21 Right eyebrow (outer)
	105,	# 22 Right eyebrow (middle)
	107,	# 23 Right eyebrow (inner)
	276,	# 24 Left eyebrow (outer)
	334,	# 25 Left eyebrow (middle)
	336,	# 26 Left eyebrow (inner)
]

mediapipe_angles = [(2, 0, 3),
                    (0, 2, 1),
                    (6, 7, 8),
                    (9, 7, 10),
                    (0, 7, 1),
                    (1, 5, 8),
                    (1, 10, 8),
                    (13, 12, 14),
                    (21, 22, 23),
                    (6, 19, 23),
                    ]

class face_prediction:
    def __init__(self, model_filename_expression):
        with open(model_filename_expression, "rb") as file:
            self.expr_model = pickle.load(file)

    def get_fts(self, face_landmarks):
        nof_fts = len(mediapipe_angles)
        fts = np.zeros(nof_fts, dtype=np.float32)
        for i in range(nof_fts):
            vector1 = face_landmarks[mediapipe_landmarks[mediapipe_angles[i][0]], :] - face_landmarks[
                                                                                       mediapipe_landmarks[
                                                                                           mediapipe_angles[i][1]], :]
            vector2 = face_landmarks[mediapipe_landmarks[mediapipe_angles[i][2]], :] - face_landmarks[
                                                                                       mediapipe_landmarks[
                                                                                           mediapipe_angles[i][1]], :]

            if sum(np.abs(np.array(vector1))) > 0 and sum(np.abs(np.array(vector2))) > 0:
                angle = vg.angle(vector1, vector2,
                                 units='rad')  # vg.signed_angle(vector1, vector2, look=vg.basis.z, units='rad')
                # angle = angle + 2 * math.pi if angle < 0 else angle
            else:
                angle = -1.0
            fts[i] = angle

        return fts
    def face_class(self, face_landmarks):
        fts = self.get_fts(face_landmarks)
        fts = fts.reshape(1, -1)
        expr_class = self.expr_model.predict(fts)
        return expr_class

