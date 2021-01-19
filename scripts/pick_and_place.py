#!/usr/bin/env python

"""
Baxter sensory data collection for Pick and Place using RSDK Inverse Kinematics

Code adapted from: https://github.com/RethinkRobotics/baxter_simulator/blob/master/baxter_sim_examples/scripts/ik_pick_and_place_demo.py
"""

import argparse
import struct
import sys
import copy
import random
import pickle
import threading
import os
import json
import time
from datetime import datetime

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from sensor_msgs.msg import Image, CompressedImage, JointState, Range, PointCloud
from rospy_message_converter import message_converter as mc 

import baxter_interface


class PickAndPlace(object):
    def __init__(self, limb, hover_distance = 0.15, verbose=True):
        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        self._gripper.calibrate()
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def pick(self, pose):
        # open the gripper
        self.gripper_open()
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()

    def place(self, pose):
        # servo above pose
        # self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()


def load_gazebo_models(table_pose=Pose(position=Point(x=1.0, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       block_pose=Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('baxter_pick_and_place_sim')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    # Load Block URDF
    block_xml = ''
    with open (model_path + "block/model.urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))


def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))


def load_gazebo_block(block_pose=Pose(position=Point(x=0.6725, y= 0.1265, z=0.7825)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('baxter_pick_and_place_sim')+"/models/"

    # Load Blocks URDF
    block_xml = ''
    block_path = "block/model.urdf"
    with open (model_path + block_path, "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
  
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))


def delete_gazebo_block():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))


def addnoise_pose(given_pose):

	new_pose = copy.deepcopy(given_pose)

	x = random.uniform(-1*0.09, 0.09)
	y = random.uniform(-1*0.09, 0.09)
	new_pose.position.x = new_pose.position.x + x
	new_pose.position.y = new_pose.position.y + y

	return new_pose


class SensorDataSaver(object):
    Recording = False
    Start_time = None
    def __init__(self):
        self._sensor_msgs = []
        self._lock = threading.Lock()

    def decode(self, sensor_msg):
        raise NotImplementedError

    def write(self, filename):
        raise NotImplementedError

    def unregister(self):
        raise NotImplementedError

    def write_and_clear(self, filename):
        filename = '{0}_{1}'.format(filename, SensorDataSaver.Start_time.strftime("%Y-%m-%d-%H-%M-%S"))
        print('writing file to {}'.format(filename), "recoding status: ", SensorDataSaver.Recording)
        self._lock.acquire()
        self._sensor_msgs = [self.decode(sensor_msg) for sensor_msg in self._sensor_msgs]
        self.write(filename)
        self._sensor_msgs *= 0 # equivalent to self._modals.clear(), python2.7 do not provide this method
        self._lock.release()

    def __del__(self):
        self.unregister()

    def callback(self, sensor_msg):
        raise NotImplementedError


class HapticDataSaver(SensorDataSaver, object):
    def __init__(self):
        super(HapticDataSaver, self).__init__()
        self.haptic_sub = rospy.Subscriber("/robot/joint_states", JointState, self.callback)

    def decode(self, sensor_msg):
        return sensor_msg

    def write(self, filename):
        with open('{}_haptic.json'.format(filename), 'wb') as f:
        	json.dump(self._sensor_msgs, f, indent=4)

    def unregister(self):
        self.haptic_sub.unregister()

    def callback(self, sensor_msg):
        self._lock.acquire()
        if SensorDataSaver.Recording:
            self._sensor_msgs.append(mc.convert_ros_message_to_dictionary(sensor_msg))
        self._lock.release()


class IRRangeDataSaver(SensorDataSaver, object):
    def __init__(self):
        super(IRRangeDataSaver, self).__init__()
        self.haptic_sub = rospy.Subscriber("/robot/range/left_hand_range/state", Range, self.callback)

    def decode(self, sensor_msg):
        return sensor_msg

    def write(self, filename):
        with open('{}_irrange.json'.format(filename), 'wb') as f:
        	json.dump(self._sensor_msgs, f, indent=4)

    def unregister(self):
        self.haptic_sub.unregister()

    def callback(self, sensor_msg):
        self._lock.acquire()
        if SensorDataSaver.Recording:
            self._sensor_msgs.append(mc.convert_ros_message_to_dictionary(sensor_msg))
        self._lock.release()


class SolarDataSaver(SensorDataSaver, object):
    def __init__(self):
        super(SolarDataSaver, self).__init__()
        self.haptic_sub = rospy.Subscriber("/robot/sonar/head_sonar/state", PointCloud, self.callback)

    def decode(self, sensor_msg):
        return sensor_msg

    def write(self, filename):
        with open('{}_solar.json'.format(filename), 'wb') as f:
        	json.dump(self._sensor_msgs, f, indent=4)

    def unregister(self):
        self.haptic_sub.unregister()

    def callback(self, sensor_msg):
        self._lock.acquire()
        if SensorDataSaver.Recording:
            self._sensor_msgs.append(mc.convert_ros_message_to_dictionary(sensor_msg))
        self._lock.release()


class ImageDataSaver(SensorDataSaver, object):
    def __init__(self):
        super(ImageDataSaver, self).__init__()
        self.image_sub = rospy.Subscriber("/cameras/head_camera/image", Image, self.callback)
        self.bridge = CvBridge()

    def decode(self, sensor_msg):
        return self.bridge.imgmsg_to_cv2(sensor_msg, sensor_msg.encoding)

    def write(self, filename):
        filename += "_vision"
        if not os.path.exists(filename):
            os.mkdir(filename)
        for i, image in enumerate(self._sensor_msgs):
            cv2.imwrite(os.path.join(filename, '{}.png'.format(i)), image)

    def unregister(self):
        self.image_sub.unregister()

    def callback(self, sensor_msg):
        self._lock.acquire()
        if SensorDataSaver.Recording:
            self._sensor_msgs.append(sensor_msg)
        self._lock.release()


def start_recording():
    SensorDataSaver.Recording = True
    SensorDataSaver.Start_time = datetime.now()


def stop_recording():
    SensorDataSaver.Recording = False


def main():

    rospy.init_node("pick_and_place")

    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_models)

    # Wait for the All Clear from emulator startup
    rospy.wait_for_message("/robot/sim/started", Empty)

    rospack = rospkg.RosPack()
    PATH = rospack.get_path('baxter_pick_and_place_sim')
    sensor_data = PATH + os.sep + "sensor_data/"
    if not os.path.exists(sensor_data):
        os.mkdir(sensor_data)

    # parse argument
    myargv = rospy.myargv(argv=sys.argv)
    num_of_run = int(myargv[1])

    limb = 'left'
    hover_distance = 0.15 # meters
    # Starting Joint angles for left arm
    starting_joint_angles = {'left_w0': 0.6699952259595108,
                             'left_w1': 1.030009435085784,
                             'left_w2': -0.4999997247485215,
                             'left_e0': -1.189968899785275,
                             'left_e1': 1.9400238130755056,
                             'left_s0': -0.08000397926829805,
                             'left_s1': -0.9999781166910306}
    pnp = PickAndPlace(limb, hover_distance)
    # An orientation for gripper fingers to be overhead and parallel to the obj
    overhead_orientation = Quaternion(
                             x=-0.0249590815779,
                             y=0.999649402929,
                             z=0.00737916180073,
                             w=0.00486450832011)

    # The Pose of the block in its initial location.
    block_pose = Pose(position= Point(x=0.7, y=0.15, z=-0.145), orientation=overhead_orientation)

    # Move to the desired starting angles
    pnp.move_to_start(starting_joint_angles)

    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    load_gazebo_models()

    haptic_saver = HapticDataSaver()
    ir_range_saver = IRRangeDataSaver()
    solar_saver = SolarDataSaver()
    image_saver = ImageDataSaver()

    for y in range(num_of_run):
		if(not rospy.is_shutdown()):
			print("\nPicking...")
			start_recording()
			fn_pick = os.path.join(sensor_data, "baxter_pick")
			pnp.pick(block_pose)
			stop_recording()
			haptic_saver.write_and_clear(fn_pick)
			ir_range_saver.write_and_clear(fn_pick)
			solar_saver.write_and_clear(fn_pick)
			image_saver.write_and_clear(fn_pick)

			print("\nPlacing...")
			start_recording()
			fn_place = os.path.join(sensor_data, "baxter_place")
			block_pose2 = addnoise_pose(block_pose)
			pnp.place(block_pose2)
			pnp.gripper_open()
			stop_recording()
			haptic_saver.write_and_clear(fn_place)
			ir_range_saver.write_and_clear(fn_place)
			solar_saver.write_and_clear(fn_place)
			image_saver.write_and_clear(fn_place)

			# pnp.move_to_start(starting_joint_angles)
			delete_gazebo_block()
			load_gazebo_block()
		else:
			break

    return 0

if __name__ == '__main__':
    sys.exit(main())
