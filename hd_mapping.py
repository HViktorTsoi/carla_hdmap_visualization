#!/usr/bin/env python
# coding=utf-8
import copy
import glob
import os
import sys

sys.path.insert(0, '/home/hvt/Code/ros_py3_ws/devel/lib/python3/dist-packages')

import carla

import argparse

import cv2
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from nav_msgs.msg import Odometry
import tf2_ros

try:
    from queue import Queue
    from queue import Empty
except ImportError:
    from Queue import Queue
    from Queue import Empty
from matplotlib import cm, pyplot as plt

import numpy as np
import numpy.linalg as LA

import open3d as o3d
import cv2 as cv
import utilities as U

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


class RosBridge:
    def __init__(self, rate=10):
        # ================================================================================================================
        # Init ros
        rospy.init_node('carla_hdmap_publish')
        rospy.loginfo('Init......')
        self.rate = rospy.Rate(rate)

        self.br = tf2_ros.TransformBroadcaster()

        self.pub_hdmap = rospy.Publisher('/HDmap', PointCloud2, queue_size=10)
        self.pub_pc = rospy.Publisher('/pc', PointCloud2, queue_size=10)
        self.pub_img = rospy.Publisher('/Image', Image, queue_size=10)
        self.pub_odom = rospy.Publisher('/Odometry', Odometry, queue_size=10)
        self.pub_info = rospy.Publisher('/Info', Image, queue_size=10)

        self.info_image = generate_info_image()

    def update_timestamp(self):
        self.timestamp = rospy.Time().to_sec()

    def publish_image(self, camera_data):
        # process camera data
        im_array = np.copy(np.frombuffer(camera_data.raw_data, dtype=np.dtype("uint8")))
        im_array = np.reshape(im_array, (camera_data.height, camera_data.width, 4))
        # im_array = im_array[:, :, :3][:, :, ::-1]
        im_array = im_array[:, :, :3]
        img_msg = ros_numpy.image.numpy_to_image(im_array, encoding='bgr8')
        img_msg.header.stamp = rospy.Time().from_sec(self.timestamp)
        self.pub_img.publish(img_msg)

    def publish_map(self, hd_map_frame, cur_frame):
        hd_msg = U.to_ros_pc2_msg(hd_map_frame, self.timestamp, frame_id='world')
        self.pub_hdmap.publish(hd_msg)

        cur_pc_msg = U.to_ros_pc2_msg(cur_frame, self.timestamp, frame_id='world')
        self.pub_pc.publish(cur_pc_msg)

    def publish_info(self):
        info_msg = ros_numpy.image.numpy_to_image(self.info_image, encoding='bgr8')
        info_msg.header.stamp = rospy.Time().from_sec(self.timestamp)
        self.pub_info.publish(info_msg)

    def publish_odom(self, T__world__o__lidar):
        # publish transform
        t = TransformStamped()
        t.header.stamp = rospy.Time().from_sec(self.timestamp)
        t.header.frame_id = 'world'
        t.child_frame_id = 'body'

        # position
        t.transform.translation = Vector3(*T__world__o__lidar[:3, 3])

        # rotation
        q = U.create_quaternion_xyzw_from_so3_matrix(T__world__o__lidar)
        t.transform.rotation = Quaternion(*q)
        self.br.sendTransform(t)

        # publish odometry
        odom = Odometry()
        odom.header.frame_id = 'world'
        odom.child_frame_id = 'body'
        odom.header = t.header
        odom.pose.pose.position = t.transform.translation
        odom.pose.pose.orientation = t.transform.rotation
        self.pub_odom.publish(odom)


def process_semantic_lidar(lidar_data, T__world__o__lidar):
    # Get the lidar data and convert it to a numpy array.
    p_cloud_size = len(lidar_data)
    # xyz is the first 3 channels
    p_cloud_xyz = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))) \
                      .reshape(p_cloud_size, -1)[:, :3]
    # semantic label is the last channel
    p_cloud_labels = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('u4'))) \
                         .reshape(p_cloud_size, -1)[:, -1]
    p_cloud = np.column_stack([
        p_cloud_xyz[:, 0],
        -p_cloud_xyz[:, 1],
        p_cloud_xyz[:, 2],
        p_cloud_labels.reshape(-1, 1)
    ])

    # point cloud distance
    dist = LA.norm(p_cloud_xyz, axis=1)
    dist_th = 30

    # filter pointcloud according to label
    hd_map_frame = p_cloud[np.where(
        (dist < dist_th) &
        (
            (p_cloud[:, -1] == 6)
            # (p_cloud[:, -1] == 17) |
            # (p_cloud[:, -1] == 11)
        )
    )]

    # useless point clouds
    cur_frame = p_cloud[np.where(
        (dist < dist_th) & (p_cloud[:, -1] != 6) & (p_cloud[:, -1] != 10) | (dist >= dist_th)
    )]

    # transform point cloud to world
    hd_map_frame[:, :3] = U.compose(T__world__o__lidar, hd_map_frame)[:, :3]
    cur_frame[:, :3] = U.compose(T__world__o__lidar, cur_frame)[:, :3]
    # offset to highlight the hd map
    cur_frame[:, 2] -= 0.5

    return hd_map_frame, cur_frame


def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)


def hd_mapping(args):
    # ================================================================================================================
    # Init ros

    ros_bridge = RosBridge()

    # ================================================================================================================
    # Init Controller

    # ================================================================================================================
    # Init Carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    # map_name = 'Town04'
    # map_name = 'Town10HD'
    map_name = 'Town05_Opt'
    # map_name = 'Town03_Opt'
    world = client.get_world()
    if map_name not in world.get_map().name:
        world = client.load_world(map_name)
    spawn_points = world.get_map().get_spawn_points()
    print(spawn_points)

    # weather
    world.set_weather(carla.WeatherParameters.WetSunset)

    bp_lib = world.get_blueprint_library()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    # 清空所有actor
    actor_list = world.get_actors()
    for actor in actor_list:
        # 检查actor是否可以被安全删除，例如，你可能不想删除某些关键的系统actor
        # 这里我们简单地尝试删除所有找到的actor
        if actor.type_id in ['vehicle.lincoln.mkz_2020', 'sensor.camera.rgb', 'sensor.lidar.ray_cast_semantic']:
            print(f'Deleting actor {actor.id} of type {actor.type_id}')
            actor.destroy()

        if isinstance(actor, carla.TrafficLight):
            # for any light, first set the light state, then set time. for yellow it is
            # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
            actor.set_state(carla.TrafficLightState.Green)
            # actor.set_red_time(5.0)
            actor.set_green_time(50000.0)
            # actor_.set_yellow_time(1000.0)

    vehicle = None
    vehicle_lidar = None

    try:
        # Search the desired blueprints
        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz_2020")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast_semantic")[0]

        # Configure the blueprints
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("shutter_speed", str(1000.0))
        camera_bp.set_attribute("motion_blur_intensity", str(0.0))
        camera_bp.set_attribute("fstop", str(5.6))
        camera_bp.set_attribute("fov", str(120))

        if args.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
        # lidar_bp.set_attribute('horizontal_fov', str(360))
        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))
        lidar_bp.set_attribute('sensor_tick', str(0.1))
        lidar_bp.set_attribute('rotation_frequency', str(10.0))

        # Spawn the blueprints
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[0])
        vehicle.set_autopilot(True)

        # set autopilot properties
        traffic_manager.auto_lane_change(vehicle, True)
        traffic_manager.random_left_lanechange_percentage(vehicle, 40)
        traffic_manager.random_right_lanechange_percentage(vehicle, 40)
        traffic_manager.keep_right_rule_percentage(vehicle, 20)

        vehicle_lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=0.8, z=1.7)),
            attach_to=vehicle
        )

        vehicle_camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=-1, y=0, z=3), carla.Rotation(roll=0, pitch=-15, yaw=0)),
            attach_to=vehicle)

        # spectator.set_transform(carla.Transform(rsu_lidar.get_transform().location, carla.Rotation(pitch=-90)))

        # The sensor data will be saved in thread-safe Queues
        lidar_queue = Queue()
        camera_queue = Queue()

        vehicle_lidar.listen(lambda data: sensor_callback(data, lidar_queue))
        vehicle_camera.listen(lambda data: sensor_callback(data, camera_queue))

        # mapping = U.to_o3d_pointcloud()
        # traj = []
        # vis = U.NonBlockVisualizer(point_size=2, origin=True)

        for frame in range(args.frames):
            ros_bridge.rate.sleep()
            world.tick()
            world_frame = world.get_snapshot().frame

            # # 相机随视角移动
            # spectator = world.get_spectator()
            # # T_cam = spectator.get_transform()
            # location = vehicle.get_transform().location
            # rotation = vehicle.get_transform().rotation
            # spectator.set_transform(carla.Transform(
            #     carla.Location(x=location.x, y=location.y, z=location.z + 3),
            #     carla.Rotation(roll=rotation.roll, pitch=rotation.pitch - 20, yaw=rotation.yaw, ),
            # ))

            try:
                # Get the data once it's received.
                lidar_data = lidar_queue.get(True, 1.0)
                while not lidar_queue.empty():
                    lidar_queue.get(True, 0.1)

                camera_data = camera_queue.get(True, 1.0)
                while not camera_queue.empty():
                    camera_queue.get(True, 0.1)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            # process semantic lidar data
            T__world__o__lidar = U.create_se3_matrix_from_carla_transformation(vehicle_lidar.get_transform())
            hd_map_frame, cur_frame = process_semantic_lidar(lidar_data, T__world__o__lidar)

            ros_bridge.update_timestamp()
            ros_bridge.publish_map(hd_map_frame, cur_frame)
            ros_bridge.publish_image(camera_data)
            ros_bridge.publish_info()
            ros_bridge.publish_odom(T__world__o__lidar)

            # cv2.imshow('camera', im_array)
            # cv2.waitKey(1)
            # # visualization
            # pcd = U.to_o3d_pointcloud(p_cloud)
            # pcd.colors = o3d.utility.Vector3dVector(
            #     plt.get_cmap('tab20')(np.int_(p_cloud[:, -1]))[:, :3])
            #
            # # transform to global frame and stack a map
            # mapping += pcd.transform(T__world__o__lidar)
            #
            # # trajectory
            # traj.append(T__world__o__lidar[:3, 3])
            # traj = traj[-2:]
            # mapping += U.to_o3d_pointcloud(np.array(traj), color=[1, 1, 1])
            #
            # T__world__o__lidar_ego = copy.copy(T__world__o__lidar)
            #
            # T__world__o__lidar_ego[:3, :3] = np.eye(3)
            # # mapping_ego = copy.copy(mapping).transform(U.INV(T__world__o__lidar_ego))
            # mapping_ego = copy.copy(mapping).transform(U.INV(T__world__o__lidar))
            #
            # # U.vis(mapping, point_size=5, origin=True)
            # # vis.update_renderer(mapping)
            # vis.update_renderer(mapping_ego)
            #
            # print(p_cloud.shape)

    finally:
        # # Apply the original settings when exiting.
        # world.apply_settings(original_settings)

        # Destroy the actors in the scene.
        if vehicle_lidar:
            vehicle_lidar.destroy()
        if vehicle:
            vehicle.destroy()


def generate_info_image():
    # 创建一个黑色背景图像
    img = np.zeros((140, 320, 3), dtype=np.uint8)

    # 定义图例项的颜色和标签
    legend_items = [
        ((255, 0, 0), "HD Map"),
        ((128, 128, 128), "Current Frame"),
        ((96, 255, 249), "Vehicle Trajectory"),
    ]

    # 设置起始位置
    start_x = 10
    start_y = 20
    offset_y = 40

    # 遍历每个图例项，绘制颜色方块和文本
    for color, label in legend_items:
        color = tuple(list(color)[::-1])
        # 绘制颜色方块
        cv2.rectangle(img, (start_x, start_y), (start_x + 20, start_y + 20), color, -1)

        # 绘制文本
        cv2.putText(img, label, (start_x + 30, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 更新下一个图例项的Y位置
        start_y += offset_y
    return img


def main():
    """Start function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor sync and projection tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=50000,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '-d', '--dot-extent',
        metavar='SIZE',
        default=2,
        type=int,
        help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=0.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-30.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='100000',
        type=int,
        help='lidar points per second (default: 100000)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.dot_extent -= 1

    try:
        hd_mapping(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
