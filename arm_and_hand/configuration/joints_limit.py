import math
import numpy as np

# ------------ HEAD --------------
head_joint1_max_velo = 6.3 * (180 / math.pi)
head_joint2_max_velo = 6.3 * (180 / math.pi)

head_joint1_max_acce = 12.0 * (180 / math.pi)
head_joint2_max_acce = 12.0 * (180 / math.pi)

head_max_velocity_container = np.array([
    head_joint1_max_velo, 
    head_joint2_max_velo], dtype=np.float64)
head_max_acce_container = np.array([
    head_joint1_max_acce, 
    head_joint2_max_acce], dtype=np.float64)

# ------------ LEFT ARM --------------
left_arm_joint1_max_velo = 1.5 * (180 / math.pi)
left_arm_joint2_max_velo = 1.5 * (180 / math.pi)
left_arm_joint3_max_velo = 2.0 * (180 / math.pi)
left_arm_joint4_max_velo = 2.0 * (180 / math.pi)
left_arm_joint5_max_velo = 2.5 * (180 / math.pi)
left_arm_joint6_max_velo = 2.5 * (180 / math.pi)

left_arm_joint1_max_acce = 3 * (180 / math.pi)
left_arm_joint2_max_acce = 3 * (180 / math.pi)
left_arm_joint3_max_acce = 4 * (180 / math.pi)
left_arm_joint4_max_acce = 4 * (180 / math.pi)
left_arm_joint5_max_acce = 6 * (180 / math.pi)
left_arm_joint6_max_acce = 6 * (180 / math.pi)

left_arm_max_velocity_container = np.array([
    left_arm_joint1_max_velo, 
    left_arm_joint2_max_velo, 
    left_arm_joint3_max_velo, 
    left_arm_joint4_max_velo, 
    left_arm_joint5_max_velo, 
    left_arm_joint6_max_velo], dtype=np.float64)
left_arm_max_acce_container = np.array([
    left_arm_joint1_max_acce, 
    left_arm_joint2_max_acce, 
    left_arm_joint3_max_acce,
    left_arm_joint4_max_acce, 
    left_arm_joint5_max_acce, 
    left_arm_joint6_max_acce], dtype=np.float64)

# ------------ RIGHT ARM --------------
right_arm_joint1_max_velo = 1.5 * (180 / math.pi)
right_arm_joint2_max_velo = 1.5 * (180 / math.pi)
right_arm_joint3_max_velo = 2.0 * (180 / math.pi)
right_arm_joint4_max_velo = 2.0 * (180 / math.pi)
right_arm_joint5_max_velo = 2.5 * (180 / math.pi)
right_arm_joint6_max_velo = 2.5 * (180 / math.pi)

right_arm_joint1_max_acce = 3 * (180 / math.pi)
right_arm_joint2_max_acce = 3 * (180 / math.pi)
right_arm_joint3_max_acce = 4 * (180 / math.pi)
right_arm_joint4_max_acce = 4 * (180 / math.pi)
right_arm_joint5_max_acce = 6 * (180 / math.pi)
right_arm_joint6_max_acce = 6 * (180 / math.pi)

right_arm_max_velocity_container = np.array([
    right_arm_joint1_max_velo, 
    right_arm_joint2_max_velo, 
    right_arm_joint3_max_velo, 
    right_arm_joint4_max_velo, 
    right_arm_joint5_max_velo, 
    right_arm_joint6_max_velo], dtype=np.float64)
right_arm_max_acce_container = np.array([
    right_arm_joint1_max_acce, 
    right_arm_joint2_max_acce, 
    right_arm_joint3_max_acce,
    right_arm_joint4_max_acce, 
    right_arm_joint5_max_acce, 
    right_arm_joint6_max_acce], dtype=np.float64)

# ------------ LEFT HAND --------------
left_hand_THUMB_joint1_max_velo = 1.9  * (180 / math.pi)
left_hand_THUMB_joint2_max_velo = 1.256 * (180 / math.pi)
left_hand_THUMB_joint3_max_velo = 1.256 * (180 / math.pi)
left_hand_INDEX_joint1_max_velo = 3.0 * (180 / math.pi)
left_hand_INDEX_joint2_max_velo = 3.0 * (180 / math.pi)
left_hand_INDEX_joint3_max_velo = 3.0 * (180 / math.pi)
left_hand_MIDDLE_joint1_max_velo = 3.0 * (180 / math.pi)
left_hand_MIDDLE_joint2_max_velo = 3.0 * (180 / math.pi)
left_hand_MIDDLE_joint3_max_velo = 3.0 * (180 / math.pi)
left_hand_RING_joint1_max_velo = 3.0 * (180 / math.pi)
left_hand_RING_joint2_max_velo = 3.0 * (180 / math.pi)
left_hand_RING_joint3_max_velo = 3.0 * (180 / math.pi)
left_hand_PINKY_joint1_max_velo = 3.0 * (180 / math.pi)
left_hand_PINKY_joint2_max_velo = 3.0 * (180 / math.pi)
left_hand_PINKY_joint3_max_velo = 3.0 * (180 / math.pi)

left_hand_THUMB_joint1_max_acce = 8.0 * (180 / math.pi)
left_hand_THUMB_joint2_max_acce = 8.0 * (180 / math.pi)
left_hand_THUMB_joint3_max_acce = 8.0 * (180 / math.pi)
left_hand_INDEX_joint1_max_acce = 8.0 * (180 / math.pi)
left_hand_INDEX_joint2_max_acce = 8.0 * (180 / math.pi)
left_hand_INDEX_joint3_max_acce = 8.0 * (180 / math.pi)
left_hand_MIDDLE_joint1_max_acce = 8.0 * (180 / math.pi)
left_hand_MIDDLE_joint2_max_acce = 8.0 * (180 / math.pi)
left_hand_MIDDLE_joint3_max_acce = 8.0 * (180 / math.pi)
left_hand_RING_joint1_max_acce = 8.0 * (180 / math.pi)
left_hand_RING_joint2_max_acce = 8.0 * (180 / math.pi)
left_hand_RING_joint3_max_acce = 8.0 * (180 / math.pi)
left_hand_PINKY_joint1_max_acce = 8.0 * (180 / math.pi)
left_hand_PINKY_joint2_max_acce = 8.0 * (180 / math.pi)
left_hand_PINKY_joint3_max_acce = 8.0 * (180 / math.pi)

left_hand_max_velocity_container = np.array([
    left_hand_THUMB_joint1_max_velo, 
    left_hand_THUMB_joint2_max_velo, 
    left_hand_THUMB_joint3_max_velo, 
    left_hand_INDEX_joint1_max_velo, 
    left_hand_INDEX_joint2_max_velo, 
    left_hand_INDEX_joint3_max_velo, 
    left_hand_MIDDLE_joint1_max_velo, 
    left_hand_MIDDLE_joint2_max_velo, 
    left_hand_MIDDLE_joint3_max_velo, 
    left_hand_RING_joint1_max_velo, 
    left_hand_RING_joint2_max_velo, 
    left_hand_RING_joint3_max_velo, 
    left_hand_PINKY_joint1_max_velo, 
    left_hand_PINKY_joint2_max_velo, 
    left_hand_PINKY_joint3_max_velo], dtype=np.float64)
left_hand_max_acce_container = np.array([
    left_hand_THUMB_joint1_max_acce,
    left_hand_THUMB_joint2_max_acce,
    left_hand_THUMB_joint3_max_acce,
    left_hand_INDEX_joint1_max_acce,
    left_hand_INDEX_joint2_max_acce,
    left_hand_INDEX_joint3_max_acce,
    left_hand_MIDDLE_joint1_max_acce,
    left_hand_MIDDLE_joint2_max_acce,
    left_hand_MIDDLE_joint3_max_acce,
    left_hand_RING_joint1_max_acce,
    left_hand_RING_joint2_max_acce,
    left_hand_RING_joint3_max_acce,
    left_hand_PINKY_joint1_max_acce,
    left_hand_PINKY_joint2_max_acce,
    left_hand_PINKY_joint3_max_acce], dtype=np.float64)