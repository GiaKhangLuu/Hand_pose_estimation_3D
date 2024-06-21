import numpy as np

def get_angles_between_joints(XYZ_landmarks, landmark_dictionary):

    # Joint 1
    #a = np.array([1, 0, 0])
    #b = XYZ_landmarks[1]
    #b = b * [1, 1, 0]

    #dot = np.sum(a * b)
    #a_norm = np.linalg.norm(a)
    #b_norm = np.linalg.norm(b)
    #cos = dot / (a_norm * b_norm)
    #angle = np.degrees(np.arccos(cos))

    #angle = 180 - angle

    #c = np.cross(b, a)
    #ref = c[-1] + 1e-9
    #signs = ref / np.absolute(ref)

    #angle *= signs

    # -------------------------

    # Joint 2 
    #a = np.array([1, 0, 0])
    #b = XYZ_landmarks[1]
    #b = b * [1, 0, 1]

    #dot = np.sum(a * b)
    #a_norm = np.linalg.norm(a)
    #b_norm = np.linalg.norm(b)
    #cos = dot / (a_norm * b_norm)
    #angle = np.degrees(np.arccos(cos))

    #angle = 180 - angle

    #c = np.cross(b, a)
    #ref = c[1] + 1e-9
    #signs = ref / np.absolute(ref)

    #angle *= signs

    # ---------------------

    # Joint 3
    a = np.array([0, 1, 0])
    b = XYZ_landmarks[landmark_dictionary.index("left wrist")] - XYZ_landmarks[landmark_dictionary.index("left elbow")]
    b = b * [0, 1, 1]
    dot = np.sum(a * b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = dot / (a_norm * b_norm)
    angle = np.degrees(np.arccos(cos))

    #c = np.cross(a, b)
    #ref = c[1] + 1e-9
    #signs = ref / np.absolute(ref)

    #print("angle: ", angle)
    
    return angle

