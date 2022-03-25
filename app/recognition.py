import dlib
from pkg_resources import resource_filename
import numpy as np

class Vision:
    def __init__(self, tolerance) -> None:
        self.face_detector = self.init_face_detector()
        self.pose_predictor = self.init_pose_predictor()
        self.face_encoder = self.init_face_encoder()
        self.tolerance = tolerance

    def init_face_encoder(self):
        model_location = resource_filename(__name__, "models/dlib_face_recognition_resnet_model_v1.dat")
        return dlib.face_recognition_model_v1(model_location)

    def init_face_detector(self):
        return dlib.get_frontal_face_detector()

    def _rect_to_css(self, rect):
        """
        Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
        :param rect: a dlib 'rect' object
        :return: a plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return rect.top(), rect.right(), rect.bottom(), rect.left()

    def _css_to_rect(self, css):
        """
        Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :return: a dlib `rect` object
        """
        return dlib.rectangle(css[3], css[0], css[1], css[2])

    def _trim_css_to_bounds(self, css, image_shape):
        """
        Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :param image_shape: numpy shape of the image array
        :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

    def face_locations(self, img, number_of_times_to_upsample=1):
        """
        Returns an array of bounding boxes of human faces in a image
        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                    deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        :return: A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        raw_face_location = self.face_detector(
            img, number_of_times_to_upsample)
        face_locations = [self._trim_css_to_bounds(self._rect_to_css(
            face), img.shape) for face in self.face_detector(img, number_of_times_to_upsample)]
        return face_locations

    def init_pose_predictor(self):
        model_location = resource_filename(__name__, "models/shape_predictor_68_face_landmarks.dat")
        return dlib.shape_predictor(model_location)

    def _raw_face_landmarks(self, face_image, face_locations=None):
        if face_locations is None:
            face_locations = self.face_detector(face_image)
        else:
            face_locations = [self._css_to_rect(face_location)
                            for face_location in face_locations]

        return [self.pose_predictor(face_image, face_location) for face_location in face_locations]

    def face_landmarks(self, face_image, face_locations=None):
        """
        Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image
        :param face_image: image to search
        :param face_locations: Optionally provide a list of face locations to check.
        :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
        :return: A list of dicts of face feature locations (eyes, nose, etc)
        """
        if face_locations is None:
                face_locations = self.face_detector(face_image)
        else:
            face_locations = [self._css_to_rect(face_location)
                            for face_location in face_locations]

        landmarks = [self.pose_predictor(face_image, face_location) for face_location in face_locations]

        landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()]
                            for landmark in landmarks]

        return [{
            "chin": points[0:17],
            "left_eyebrow": points[17:22],
            "right_eyebrow": points[22:27],
            "nose_bridge": points[27:31],
            "nose_tip": points[31:36],
            "left_eye": points[36:42],
            "right_eye": points[42:48],
            "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
            "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
        } for points in landmarks_as_tuples]

    def init_face_encoder(self):
        model_location = resource_filename(__name__, "models/dlib_face_recognition_resnet_model_v1.dat")
        return dlib.face_recognition_model_v1(model_location)

    def face_encodings(self, face_image, known_face_locations=None, num_jitters=1):
        """
        Given an image, return the 128-dimension face encoding for each face in the image.
        :param face_image: The image that contains one or more faces
        :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
        :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
        :param model: Optional - which model to use. "large" or "small" (default) which only returns 5 points but is faster.
        :return: A list of 128-dimensional face encodings (one for each face in the image)
        """
        raw_landmarks = self._raw_face_landmarks(
            face_image, known_face_locations)
        return [np.array(self.face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

    def face_distance(self, face_encodings, face_to_compare):
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.
        :param face_encodings: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
        if len(face_encodings) == 0:
            return np.empty((0))

        return np.linalg.norm(face_encodings - face_to_compare, axis=1)


    def compare_faces(self, known_face_encodings, face_encoding_to_check):
        """
        Compare a list of face encodings against a candidate encoding to see if they match.
        :param known_face_encodings: A list of known face encodings
        :param face_encoding_to_check: A single face encoding to compare against the list
        :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
        """
        return list(self.face_distance(known_face_encodings, face_encoding_to_check) <= self.tolerance)