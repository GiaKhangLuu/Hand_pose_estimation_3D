from abc import ABC, abstractmethod
from angle_calculation_utilities import calculate_the_next_two_joints_angle

NUM_ANGLES_EACH_CHAIN = 2

class ChainAngleCalculator(ABC):
    def __init__(self, num_chain, landmark_dictionary):
        """
        Each chain has `NUM_ANGLES_EACH_CHAIN` (2) angles. The first angle always rotates
        about the z-axis of its parent coordinate and the second angle rotates about the 
        y-axis of the first angle coordinate. The coordinate of the second angle will be 
        a parent coordinate of a next chain's first angle.
        """
        self.num_angles_each_chain = NUM_ANGLES_EACH_CHAIN
        self.num_chain = num_chain
        self._landmark_dictionary = landmark_dictionary

        assert len(self.landmarks_name) == self.num_chain
        assert len(self._mapping_to_robot_angle_func_container) == self.num_chain
        assert len(self._vector_landmark_in_previous_frame_container) == self.num_chain
        assert len(self.rot_mat_to_rearrange_container) == self.num_chain
        assert len(self._angle_range_of_two_joints_container) == self.num_chain
        assert len(self._axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints_container) == self.num_chain
        assert len(self._get_the_opposite_of_two_joints_flag_container) == self.num_chain
        assert len(self._limit_angle_of_two_joints_flag_container) == self.num_chain
        assert len(self.calculate_second_angle_flag_container) == self.num_chain

    def _update_vector_in_previous_frame(self, chain_idx, vector_in_current_frame):
        self._vector_landmark_in_previous_frame_container[chain_idx] = vector_in_current_frame

    def _calculate_chain_angles(self, XYZ_landmarks, parent_coordinate):
        """
        To calculate a couple angles (a chain) for each joint, we need THREE values:
            1. A vector which acts as an x-vector in order to create a
                child coordinate.
            2. A rotation matrices which helps to rearrange the x-axis, 
                y-axis and z-axis into the new ones. This new coordinate
                ensures that the first joint rotates about the z-axis
                and the second joint rotates about the y-axis.
            3. Two mapping function. Each angle differs from an angles
                in robot, therefore, we have to transform the angle of
                real person into the angle of robot.
        """

        result_dict = dict()
        parent_coordinate = parent_coordinate.copy()
        for chain_idx in range(self.num_chain):
            vector_landmark = self._get_landmark_vector(chain_idx, XYZ_landmarks)
            mapping_to_robot_angle_functions = self._mapping_to_robot_angle_func_container[chain_idx]
            vector_landmark_in_previou_frame = self._vector_landmark_in_previous_frame_container[chain_idx]
            rot_mat_to_rearrange = self.rot_mat_to_rearrange_container[chain_idx]
            angle_range_of_two_joints = self._angle_range_of_two_joints_container[chain_idx]
            axis_to_get_the_opposite = self._axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints_container[chain_idx]
            get_the_opposite_of_two_joints_flag = self._get_the_opposite_of_two_joints_flag_container[chain_idx]
            limit_angle_of_two_joints_flag = self._limit_angle_of_two_joints_flag_container[chain_idx]
            calculate_second_angle_flag = self.calculate_second_angle_flag_container[chain_idx]
            
            result_of_chain = calculate_the_next_two_joints_angle(
                vector_landmark=vector_landmark,
                map_to_robot_angle_funcs=mapping_to_robot_angle_functions,
                parent_coordinate=parent_coordinate,
                vector_in_prev_frame=vector_landmark_in_previou_frame,
                rotation_matrix_to_rearrange_coordinate=rot_mat_to_rearrange,
                angle_range_of_two_joints=angle_range_of_two_joints,
                axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints=axis_to_get_the_opposite,
                get_the_opposite_of_two_joints=get_the_opposite_of_two_joints_flag,
                limit_angle_of_two_joints=limit_angle_of_two_joints_flag,
                calculate_the_second_joint=calculate_second_angle_flag 
            )
            
            if self.calculate_second_angle_flag_container[chain_idx]:
                parent_coordinate = result_of_chain["younger_brother_rot_mat_wrt_origin"].copy()
            else:
                parent_coordinate = result_of_chain["older_brother_rot_mat_wrt_origin"].copy()
            result_dict[f"chain_{chain_idx+1}"] = result_of_chain

        return result_dict

    @abstractmethod
    def _get_landmark_vector(self, chain_idx, XYZ_landmarks):
        """
        TODO: Doc.
        Get vector based on landmark_name
        Input:
            chain_idx:
            XYZ_landmarks:
        Output:
            landmark_vec (np.array): shape = (3,)
        """
        pass