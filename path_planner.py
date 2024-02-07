#!/usr/bin/env python3

from enum import Enum, auto

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scenario_msgs.msg import Viewpoints, Viewpoint
from scenario_msgs.srv import MoveToStart, SetPath
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker
from random import uniform


class State(Enum):
    UNSET = auto()
    INIT = auto()
    IDLE = auto()
    MOVE_TO_START = auto()
    NORMAL_OPERATION = auto()


def occupancy_grid_to_matrix(grid: OccupancyGrid):
    data = np.array(grid.data, dtype=np.uint8)

    data = data.reshape(grid.info.height, grid.info.width)          #getauscht
    return data


def world_to_matrix(x, y, grid_size):
    return [round(x / grid_size), round(y / grid_size)]


def matrix_index_to_world(x, y, grid_size):
    return [x * grid_size, y * grid_size]


def multiple_matrix_indeces_to_world(points, grid_size):
    world_points = []
    for point in points:
        world_points.append([point[0] * grid_size, point[1] * grid_size])
    return world_points


def compute_discrete_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    x = x0
    y = y0
    points = []
    while True:
        points.append([int(x), int(y)])
        if x == x1 and y == y1:
            break
        doubled_error = 2 * error
        if doubled_error >= dy:
            if x == x1:
                break
            error += dy
            x += sx
        if doubled_error <= dx:
            if y == y1:
                break
            error += dx
            y += +sy
    return points


class PathPlanner(Node):

    def __init__(self):
        super().__init__(node_name='path_planner')
        self.state = State.UNSET
        self.cell_size = 0.2
        self.recomputation_required = True
        self.target_viewpoint_index = -1
        self.path_marker: Marker
        self.init_path_marker()
        self.path_marker_pub = self.create_publisher(Marker, '~/marker', 1)
        self.viewpoints = []
        self.waypoints = []
        self.orientations = []
        self.occupancy_grid: OccupancyGrid = None
        self.occupancy_matrix: np.ndarray = None
        self.progress = -1.0
        self.remaining_segments = []
        self.init_clients()
        self.init_services()
        self.grid_map_sub = self.create_subscription(OccupancyGrid,
                                                     'occupancy_grid',
                                                     self.on_occupancy_grid, 1)
        self.viewpoints_sub = self.create_subscription(Viewpoints, 'viewpoints',
                                                       self.on_viewpoints, 1)

    def init_services(self):
        self.move_to_start_service = self.create_service(
            MoveToStart, '~/move_to_start', self.serve_move_to_start)
        self.start_service = self.create_service(Trigger, '~/start',
                                                 self.serve_start)
        self.stop_service = self.create_service(Trigger, '~/stop',
                                                self.serve_stop)

    def init_clients(self):
        cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.set_path_client = self.create_client(SetPath,
                                                  'path_follower/set_path',
                                                  callback_group=cb_group)
        self.path_finished_client = self.create_client(
            Trigger, 'path_follower/path_finished', callback_group=cb_group)

    def serve_move_to_start(self, request, response):
        self.state = State.MOVE_TO_START
        self.start_pose = request.target_pose
        self.current_pose = request.current_pose
        # we do not care for collisions while going to the start position
        # in the simulation 'collisions' do not matter. In the lab, we
        # can manually make sure that we avoid collisions, by bringing the
        # vehicle in a safe position manually before starting anything.
        response.success = self.move_to_start(request.current_pose,
                                              request.target_pose)
        return response

    def move_to_start(self, p0: Pose, p1: Pose):
        path_segment = self.compute_simple_path_segment(p0,
                                                        p1,
                                                        check_collision=False)
        request = SetPath.Request()
        request.path = path_segment['path']
        answer = self.set_path_client.call(request)
        if answer.success:
            self.get_logger().info('Moving to start position')
            return True
        else:
            self.get_logger().info(
                'Asked to move to start position. '
                'But the path follower did not accept the new path.')
            return False

    def has_collisions(self, points_2d):
        if not self.occupancy_grid:
            return []
        collision_indices = [
            i for i, p in enumerate(points_2d)
            if self.occupancy_matrix[p[1], p[0]] >= 50    #getauscht
        ]
        return collision_indices
    
    def find_neighbors(self, point_2d):
        neighbors = list()
        x = point_2d[0]
        y = point_2d[1]

        max_xindex = self.occupancy_matrix.shape[1] - 1
        max_yindex = self.occupancy_matrix.shape[0] - 1
        if x == 0:
            if y == 0:
                neighbors = [[x+1, y], [x, y+1], [x+1, y+1]]
            elif y == max_yindex:
                neighbors = [[x+1, y-1], [x, y-1], [x+1,y]]
            else:
                neighbors = [[x,y-1], [x, y+1], [x+1, y-1], [x+1, y], [x+1, y+1]]
            return neighbors
        elif x == max_xindex:
            if y == 0:
                neighbors = [[x-1, y], [x-1, y+1], [x, y+1]]
            elif y == max_yindex:
                neighbors = [[x, y-1], [x-1, y-1], [x-1, y]]
            else:
                neighbors = [[x, y-1], [x-1, y-1], [x-1, y], [x-1, y+1], [x, y+1]]
            return neighbors
        if y == 0:
            neighbors = [[x-1, y], [x-1, y+1], [x, y+1], [x+1, y+1], [x+1, y]]
            return neighbors
        elif y == max_yindex:
            neighbors = [[x-1, y], [x+1, y], [x+1, y-1], [x, y-1], [x-1, y-1]]
            return neighbors
        neighbors = [[x-1, y+1], [x, y+1], [x+1, y+1], [x+1, y], [x+1, y-1],[x-1, y-1], [x, y-1], [x-1, y]]
        return neighbors
    
    def compute_distance_2d_points(self, p0_2d, p1_2d):
        p0_world = matrix_index_to_world(p0_2d[0], p0_2d[1], self.cell_size)
        p1_world = matrix_index_to_world(p1_2d[0], p1_2d[1], self.cell_size)
        d = np.sqrt((p0_world[0] - p1_world[0])**2 + (p0_world[1] - p1_world[1])**2)
        return d

    def find_nearest_rrt(self, p0, nodes):
        closest_so_far = [[None, None], np.inf]
        for node in nodes:
            possible_distance = self.distance_rrt(p0, node)
            if possible_distance < closest_so_far[1]:
                closest_so_far = [node, possible_distance]
        return closest_so_far 

    def find_rrt_inbetween(self, p0, p1, d):
        v_norm = 1/np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        x_new = p0[0] + d * v_norm * (p1[0] - p0[0])
        y_new = p0[1] + d * v_norm * (p1[1] - p0[1])
        return [x_new, y_new]


    def compute_a_star_segment(self, p0: Pose, p1: Pose):
        i=0
        #self.get_logger().info('new')
        p0_2d = world_to_matrix(p0.position.x, p0.position.y, self.cell_size)
        p1_2d = world_to_matrix(p1.position.x, p1.position.y, self.cell_size)
        #self.get_logger().info(f'p0: {p0_2d}, p0_3d: {p0.position.x}, {p0.position.y}')
        path_list = list()
        list_open = list()
        list_closed = list()
        f_cost_matrix = np.inf * np.copy(self.occupancy_matrix)
        g_cost_matrix = np.inf * np.copy(self.occupancy_matrix)
        predecessor_matrix = [[[None, None] for x in range(self.occupancy_matrix.shape[1])] for y in range(self.occupancy_matrix.shape[0])]#weird, mal gucken
        #self.get_logger().info(f'{f_cost_matrix.shape}, {g_cost_matrix.shape}, {len(predecessor_matrix)}, {len(predecessor_matrix[0])}')
        list_open.append(p0_2d)
        f_cost_matrix[p0_2d[1], p0_2d[0]] = self.compute_distance_2d_points(p0_2d, p1_2d) #alle f costs auf inf außer startknoten
        g_cost_matrix[p0_2d[1], p0_2d[0]] = 0.0                      #ein Matrix Element ist eine Liste und beinhaltet
        #self.get_logger().info(f'lmao{list_open}')
        while True:
            i+=1
            #self.get_logger().info(f'{i}')
            #current_smallest_f = [None, np.inf]                                     #knoten aus open mit kleinster f finden
            #if len(list_open) == 0:
             #   break
            for i, node in enumerate(list_open):
                if i == 0:
                    current_smallest_f = [node, f_cost_matrix[node[1], node[0]]]                                                   #
                if f_cost_matrix[node[1], node[0]] < current_smallest_f[1]:         #
                    current_smallest_f = [node, f_cost_matrix[node[1], node[0]]]
            current_node = current_smallest_f[0]
            #self.get_logger().info(f'cn{current_node}')
            #self.get_logger().info(f'xd{list_open}')
            #self.get_logger().info(f'goal{p1_2d}')
            #self.get_logger().info(f'closed: {list_closed}, {len(list_closed)}')
            list_open.remove(current_node)
            list_closed.append(current_node)
            
            if current_node != p0_2d:
                current_predecessor = predecessor_matrix[current_node[1]][current_node[0]]
                g_pre = g_cost_matrix[current_predecessor[1], current_predecessor[0]]
                distance_predecessor_current = self.compute_distance_2d_points(current_predecessor, current_node)
                g_cost_matrix[current_node[1], current_node[0]] = g_pre + distance_predecessor_current
            if current_node == p1_2d:
                #self.get_logger().info('so far so good')
                break
            neighbors = self.find_neighbors(current_node)
            for i in sorted(self.has_collisions(neighbors), reverse=True):               #knoten die in Hindernissen sind werden 
                neighbors.pop(i)                                                  #nicht beachtet
            for node in neighbors:
                if node in list_closed:                                           
                    continue
                if not node in list_open:
                   g_predecessor = g_cost_matrix[current_node[1], current_node[0]]
                   f_cost_matrix[node[1], node[0]] = g_predecessor + self.compute_distance_2d_points(current_node, node)\
                                                    +self.compute_distance_2d_points(node, p1_2d)
                   predecessor_matrix[node[1]][node[0]] = current_node
                   list_open.append(node)
                else:
                    potential_f = g_cost_matrix[current_node[1], current_node[0]]\
                                + self.compute_distance_2d_points(current_node, node)\
                                + self.compute_distance_2d_points(node, p1_2d)
                    if potential_f < f_cost_matrix[node[1], node[0]]:
                        f_cost_matrix[node[1], node[0]] = potential_f
                        predecessor_matrix[node[1]][node[0]] = current_node
        
        node_to_add = p1_2d
        i = 0
        while True:
            path_list.insert(0, node_to_add)
            if node_to_add == p0_2d:
                break
            node_to_add = predecessor_matrix[node_to_add[1]][node_to_add[0]]

        #self.get_logger().info(f'{path_list}')
        #self.get_logger().info('jawohl')
        xy_3d = multiple_matrix_indeces_to_world(path_list, self.cell_size)
        #self.get_logger().info(f'{xy_3d}')
        if len(xy_3d) == 1:
            xy_3d.append(xy_3d[0])
        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(xy_3d) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(xy_3d)
        ]
        # Replace the last point with the exac value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position
        # Now we have a waypoint path with the x and y component computed by
        # our path finding algorithm and z is a linear interpolation between
        # the z coordinate of the start and the goal pose.

        # now we need to compute our desired heading (yaw angle) while we
        # follow the waypoints. We choose a not-so-clever approach by
        # keeping the yaw angle from our start pose and only set the yaw
        # angle to the desired yaw angle from the goal pose for the very last
        # waypoint
        q0 = p0.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
        q1 = p1.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])

        # replace the very last orientation with the orientation of our
        # goal pose p1.
        #q = quaternion_from_euler(0.0, 0.0, yaw0)
        q = quaternion_from_euler(0.0, 0.0, yaw1)#yaw0
        orientations = [Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                        ] * len(points_3d)
        q = quaternion_from_euler(0.0, 0.0, yaw1)
        orientations[-1] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        #self.get_logger().info('almost done')
        return {'path': path, 'collision_indices': []}
        
    def check_rrt_vortex_for_collision(self, p0, p1):
        p0_2d = world_to_matrix(p0[0], p0[1], self.cell_size)
        p1_2d = world_to_matrix(p1[0], p1[1], self.cell_size)
        line_points = compute_discrete_line(p0_2d[0], p0_2d[1], p1_2d[0], p1_2d[1])
        num_collisions = len(self.has_collisions(line_points))
        if num_collisions == 0:
            return False
        else:
            return True                                                

    def check_rrt_vortex2(self, p0, p1, num_points):
        x0 = p0[0]
        y0 = p0[1]
        x1 = p1[0]
        y1 = p1[1]
        x_list = np.linspace(x0, x1, num_points)
        y_list = np.linspace(y0, y1, num_points)
        matrix_point_list = list()
        for i in range(num_points):
            matrix_point_list.append(world_to_matrix(x_list[i], y_list[i], self.cell_size))
        if len(self.has_collisions(matrix_point_list)) != 0:
            return True
        else:
            return False
        
    def check_rrt_point(self, p0):
        p_matrix = world_to_matrix(p0[0], p0[1], self.cell_size)
        if len(self.has_collisions([p_matrix])) != 0:
            return True
        else:
            return False

    def distance_rrt(self, p0, p1):
        return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
    
    def neighbors_rrt(self, radius, p, nodes):
        neighbors = []
        for candidate in nodes:
            if self.distance_rrt(p, candidate) < radius:
                neighbors.append(candidate)
        return neighbors

    def rrt_update_cost(self, parent, children_list, children_list_global, cost_to_node_list, node_list, delta_cost):
        parent_index = node_list.index(parent)
        cost_to_node_list[parent_index] -= delta_cost
        if len(children_list) == 0: 
            return
        for child in children_list:
            new_children_list = children_list_global[node_list.index(child)]
            self.rrt_update_cost(child, new_children_list, children_list_global, cost_to_node_list, node_list, delta_cost) 
        

    def compute_rrt_star_segment(self, p0: Pose, p1: Pose):
        p_start = [p0.position.x, p0.position.y]
        p_finish = [p1.position.x, p1.position.y]
        node_list = [p_start]
        predecessor_list = [None]
        cost_to_node_list = [0]
        iterations = 100  #600
        radius = 1.0      #0.3
        i=0
        loop_iterations = 0
        max_distance_to_goal = 0.4    #0.15
        current_distance_to_goal = np.inf#######
        goal_reached = False
        path_list = []
        while i < iterations:
            loop_iterations+=1
            candidate = [uniform(0.05, 1.95), uniform(0.05, 3.95)]
            if self.check_rrt_point(candidate) == True:
                continue
            neighbors = self.neighbors_rrt(radius, candidate, node_list)
            if len(neighbors) == 0:
                continue
            current_parent_and_distance = [None, np.inf]
            for k, possible_parent in enumerate(neighbors):
                current_weight = cost_to_node_list[node_list.index(possible_parent)] + self.distance_rrt(possible_parent, candidate)
                if current_weight < current_parent_and_distance[1]:
                    #if self.check_rrt_vortex_for_collision(candidate, possible_parent) == False:
                    if self.check_rrt_vortex2(candidate, possible_parent, 20) == False:
                        current_parent_and_distance = [possible_parent, current_weight]  
                    else:
                        continue
            if current_parent_and_distance[0] == None:
                continue
            node_list.append(candidate)
            predecessor_list.append(current_parent_and_distance[0])
            cost_to_node_list.append(current_parent_and_distance[1])
            for to_be_improved in neighbors:
                index_of_to_be_improved = node_list.index(to_be_improved)
                new_cost = cost_to_node_list[-1] + self.distance_rrt(candidate, to_be_improved)
                if new_cost < cost_to_node_list[index_of_to_be_improved]:
                    #if self.check_rrt_vortex_for_collision(candidate, to_be_improved) == False:
                    if self.check_rrt_vortex2(candidate, to_be_improved, 20) == False:
                        predecessor_list[index_of_to_be_improved] = candidate
                        cost_to_node_list[index_of_to_be_improved] = new_cost
                    else:
                        continue
            candidate_distance_to_goal = self.distance_rrt(candidate, p_finish)
            if candidate_distance_to_goal < max_distance_to_goal and candidate_distance_to_goal< current_distance_to_goal\
                and self.check_rrt_vortex2(candidate, p_finish, 20) == False:
                #and self.check_rrt_vortex_for_collision(candidate, p_finish) == False:
                
                
                goal_reached = True
                goal_coordinates = candidate
                current_distance_to_goal = candidate_distance_to_goal
            i+=1
        current_path_length = cost_to_node_list[node_list.index(goal_coordinates)]
            
        if goal_reached == False:
            #self.get_logger().info('Goal not found')
            return 0
        node_to_be_added = goal_coordinates
        while not p_start in path_list:
            path_list.insert(0, node_to_be_added)
            node_to_be_added = predecessor_list[node_list.index(node_to_be_added)]
        xy_3d = path_list
        #self.get_logger().info('at important point')
        #elf.get_logger().info(f'p0: {p_start}, p1: {p1}, list: {xy_3d}')
        
        if len(xy_3d) == 1:
            xy_3d.append(xy_3d[0])

        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(xy_3d) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(xy_3d)
        ]
        # Replace the last point with the exac value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position
        # Now we have a waypoint path with the x and y component computed by
        # our path finding algorithm and z is a linear interpolation between
        # the z coordinate of the start and the goal pose.

        # now we need to compute our desired heading (yaw angle) while we
        # follow the waypoints. We choose a not-so-clever approach by
        # keeping the yaw angle from our start pose and only set the yaw
        # angle to the desired yaw angle from the goal pose for the very last
        # waypoint
        q0 = p0.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
        q1 = p1.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])

        # replace the very last orientation with the orientation of our
        # goal pose p1.
        #q = quaternion_from_euler(0.0, 0.0, yaw0)
        q = quaternion_from_euler(0.0, 0.0, yaw1)#yaw0
        orientations = [Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                        ] * len(points_3d)
        q = quaternion_from_euler(0.0, 0.0, yaw1)
        orientations[-1] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.get_logger().info(f'path length = {current_path_length}')
        self.get_logger().info(f'loop iterations: {loop_iterations}')

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        #self.get_logger().info('almost done')
        return [{'path': path, 'collision_indices': []}, current_path_length]
            
    def compute_rrt_star_segment2(self, p0: Pose, p1: Pose):
        p_start = [p0.position.x, p0.position.y]
        p_finish = [p1.position.x, p1.position.y]
        node_list = [p_start]
        predecessor_list = [None]
        cost_to_node_list = [0]
        iterations = 100  #600
        radius = 1.0      #0.3
        d = 0.9 * radius
        i=0
        loop_iterations = 0
        max_distance_to_goal = 0.4    #0.15
        current_distance_to_goal = np.inf#######
        goal_reached = False
        path_list = []
        while i < iterations:
            loop_iterations+=1
            candidate = [uniform(0.05, 1.95), uniform(0.05, 3.95)]
            closest_node, distance_to_closest = self.find_nearest_rrt(candidate, node_list)
            if distance_to_closest > d:
                candidate = self.find_rrt_inbetween(closest_node, candidate, d)
            if self.check_rrt_point(candidate) == True:
                continue
            neighbors = self.neighbors_rrt(radius, candidate, node_list)
            if len(neighbors) == 0:
                continue
            current_parent_and_distance = [None, np.inf]
            for k, possible_parent in enumerate(neighbors):
                current_weight = cost_to_node_list[node_list.index(possible_parent)] + self.distance_rrt(possible_parent, candidate)
                if current_weight < current_parent_and_distance[1]:
                    #if self.check_rrt_vortex_for_collision(candidate, possible_parent) == False:
                    if self.check_rrt_vortex2(candidate, possible_parent, 20) == False:
                        current_parent_and_distance = [possible_parent, current_weight]  
                    else:
                        continue
            if current_parent_and_distance[0] == None:
                continue
            node_list.append(candidate)
            predecessor_list.append(current_parent_and_distance[0])
            cost_to_node_list.append(current_parent_and_distance[1])
            for to_be_improved in neighbors:
                index_of_to_be_improved = node_list.index(to_be_improved)
                new_cost = cost_to_node_list[-1] + self.distance_rrt(candidate, to_be_improved)
                if new_cost < cost_to_node_list[index_of_to_be_improved]:
                    #if self.check_rrt_vortex_for_collision(candidate, to_be_improved) == False:
                    if self.check_rrt_vortex2(candidate, to_be_improved, 20) == False:
                        predecessor_list[index_of_to_be_improved] = candidate
                        cost_to_node_list[index_of_to_be_improved] = new_cost
                    else:
                        continue
            candidate_distance_to_goal = self.distance_rrt(candidate, p_finish)
            if candidate_distance_to_goal < max_distance_to_goal and candidate_distance_to_goal< current_distance_to_goal\
                and self.check_rrt_vortex2(candidate, p_finish, 20) == False:
                #and self.check_rrt_vortex_for_collision(candidate, p_finish) == False:
                
                
                goal_reached = True
                goal_coordinates = candidate
                current_distance_to_goal = candidate_distance_to_goal
            i+=1
        current_path_length = cost_to_node_list[node_list.index(goal_coordinates)]
            
        if goal_reached == False:
            #self.get_logger().info('Goal not found')
            return 0
        node_to_be_added = goal_coordinates
        while not p_start in path_list:
            path_list.insert(0, node_to_be_added)
            node_to_be_added = predecessor_list[node_list.index(node_to_be_added)]
        xy_3d = path_list
        #self.get_logger().info('at important point')
        #elf.get_logger().info(f'p0: {p_start}, p1: {p1}, list: {xy_3d}')
        
        if len(xy_3d) == 1:
            xy_3d.append(xy_3d[0])

        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(xy_3d) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(xy_3d)
        ]
        # Replace the last point with the exac value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position
        # Now we have a waypoint path with the x and y component computed by
        # our path finding algorithm and z is a linear interpolation between
        # the z coordinate of the start and the goal pose.

        # now we need to compute our desired heading (yaw angle) while we
        # follow the waypoints. We choose a not-so-clever approach by
        # keeping the yaw angle from our start pose and only set the yaw
        # angle to the desired yaw angle from the goal pose for the very last
        # waypoint
        q0 = p0.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
        q1 = p1.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])

        # replace the very last orientation with the orientation of our
        # goal pose p1.
        #q = quaternion_from_euler(0.0, 0.0, yaw0)
        q = quaternion_from_euler(0.0, 0.0, yaw1)#yaw0
        orientations = [Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                        ] * len(points_3d)
        q = quaternion_from_euler(0.0, 0.0, yaw1)
        orientations[-1] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.get_logger().info(f'path length = {current_path_length}')
        self.get_logger().info(f'loop iterations: {loop_iterations}')

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        #self.get_logger().info('almost done')
        return [{'path': path, 'collision_indices': []}, current_path_length]

    def compute_rrt_star_segment22(self, p0: Pose, p1: Pose):
        #benötigte Funktionen:
        #neighbors_rrt
        #distance_rrt
        #check_rrt_point
        #check_rrt_vortex2
        #find_rrt_inbetween
        #find_nearest_rrt
        p_start = [p0.position.x, p0.position.y]
        p_finish = [p1.position.x, p1.position.y]
        node_list = [p_start]
        predecessor_list = [None]       #wichtig um später den Pfad zu rekonstruieren
        global_children_list = [[]]     #wichtig um die Kostenfunktion upzudaten
        cost_to_node_list = [0]
        iterations = 100  #600
        radius = 1.0      #0.3
        d = 0.9 * radius
        i=0
        loop_iterations = 0
        max_distance_to_goal = 0.4    #0.15
        current_distance_to_goal = np.inf#######
        goal_reached = False
        path_list = []

        while i < iterations:
            loop_iterations+=1
            candidate = [uniform(0.05, 1.95), uniform(0.05, 3.95)]
            closest_node, distance_to_closest = self.find_nearest_rrt(candidate, node_list)

            if distance_to_closest > d:
                candidate = self.find_rrt_inbetween(closest_node, candidate, d)
            #sorgt dafür, dass jeder Knoten mindestens einen Nachbarn hat, in dem der Knoten bei zu großer
            #Entfernung zum nächsten Knoten an diesen "herangerückt" wird

            if self.check_rrt_point(candidate) == True: #prüft, ob der neue Knoten in Obstacle ist
                continue
            neighbors = self.neighbors_rrt(radius, candidate, node_list)
            if len(neighbors) == 0:    
                continue
            current_parent_and_distance = [None, np.inf]
            
            for k, possible_parent in enumerate(neighbors):
                current_weight = cost_to_node_list[node_list.index(possible_parent)] + self.distance_rrt(possible_parent, candidate)
                if current_weight < current_parent_and_distance[1]:
                    if self.check_rrt_vortex2(candidate, possible_parent, 20) == False:
                        current_parent_and_distance = [possible_parent, current_weight]  
                    else:
                        continue  
            #findet aus den Nachbarn den Knoten, der die Distanz zum Startknoten minimiert als Vorgänger

            if current_parent_and_distance[0] == None:
                continue
            #jeder Knoten, der in die Liste kommt, soll einen Vorgänger haben

            node_list.append(candidate)
            predecessor_list.append(current_parent_and_distance[0])
            cost_to_node_list.append(current_parent_and_distance[1])
            global_children_list.append([])
            global_children_list[node_list.index(current_parent_and_distance[0])].append(candidate)

            for to_be_improved in neighbors:
                index_of_to_be_improved = node_list.index(to_be_improved)
                new_cost = cost_to_node_list[-1] + self.distance_rrt(candidate, to_be_improved)
                if new_cost < cost_to_node_list[index_of_to_be_improved]:
                    if self.check_rrt_vortex2(candidate, to_be_improved, 20) == False:
                        delta_cost = cost_to_node_list[index_of_to_be_improved] - new_cost
                        global_children_list[node_list.index(predecessor_list[index_of_to_be_improved])].remove(to_be_improved)
                        global_children_list[-1].append(to_be_improved)
                        predecessor_list[index_of_to_be_improved] = candidate
                        self.rrt_update_cost(to_be_improved, global_children_list[index_of_to_be_improved], global_children_list, cost_to_node_list, node_list, delta_cost)
                    else:
                        continue
            #prüft für die Nachbarknoten, ob der neue Knoten ein besserer Vorgänger wäre, indem geschaut wird, 
            #ob die Kostenfunktion mit dem neuen Knoten als Vorgänger kleiner ist als die aktueller Kostenfunktion 
            #und ob die Kante nicht durch Obstacles geht. Dann wird der neue Knoten als Vorgänger gesetzt, beim neuen Knoten
            #der Nachbar als Nachfolger hinzugefügt und beim vorherigen Vorgänger der Nachbar als Nachfolger enfernt. 
            #Die Kostenfunktion des Nachbarn wird rekursiv geupdated(der Knoten selbst und die Nachfoler, die Nachfolger 
            #der Nachfolger usw. )

            candidate_distance_to_goal = self.distance_rrt(candidate, p_finish)
            if candidate_distance_to_goal < max_distance_to_goal and candidate_distance_to_goal< current_distance_to_goal\
                and self.check_rrt_vortex2(candidate, p_finish, 20) == False:
                goal_coordinates = candidate
                current_distance_to_goal = candidate_distance_to_goal
                goal_reached = True
            #ein Knoten, der eine Maximaldistanz zum Ziel unterschreitet, näher dran ist als ein evtl. vorher gefundener
            #vorletzter Knoten und dessen Kante zum Ziel nicht durch Obstacles geht wird als neuer vorletzter markiert

            i+=1
        current_path_length = cost_to_node_list[node_list.index(goal_coordinates)] + self.distance_rrt(goal_coordinates, p_finish)
            
        if goal_reached == False:
            self.get_logger().info('Goal not found')
            return 0
        
        node_to_be_added = goal_coordinates
        while not p_start in path_list:
            path_list.insert(0, node_to_be_added)
            node_to_be_added = predecessor_list[node_list.index(node_to_be_added)]
        #Es wird der vorletzte Knoten eingefügt und dann immer am Anfang der Liste der Vorgänger, bis der Startknoten erreicht ist

        xy_3d = path_list
        
        if len(xy_3d) == 1:
            xy_3d.append(xy_3d[0])

        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(xy_3d) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(xy_3d)
        ]
        # Replace the last point with the exac value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position
        # Now we have a waypoint path with the x and y component computed by
        # our path finding algorithm and z is a linear interpolation between
        # the z coordinate of the start and the goal pose.

        # now we need to compute our desired heading (yaw angle) while we
        # follow the waypoints. We choose a not-so-clever approach by
        # keeping the yaw angle from our start pose and only set the yaw
        # angle to the desired yaw angle from the goal pose for the very last
        # waypoint
        q0 = p0.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
        q1 = p1.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])

        # replace the very last orientation with the orientation of our
        # goal pose p1.
        #q = quaternion_from_euler(0.0, 0.0, yaw0)
        q = quaternion_from_euler(0.0, 0.0, yaw1)#yaw0
        orientations = [Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                        ] * len(points_3d)
        q = quaternion_from_euler(0.0, 0.0, yaw1)
        orientations[-1] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.get_logger().info(f'path length = {current_path_length}')
        self.get_logger().info(f'loop iterations: {loop_iterations}')

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        #self.get_logger().info('almost done')
        return [{'path': path, 'collision_indices': []}, current_path_length]

    def compute_rrt_star_segment3(self, p0: Pose, p1: Pose):
        p_start = [p0.position.x, p0.position.y]
        p_finish = [p1.position.x, p1.position.y]
        node_list = [p_start]
        predecessor_list = [None]
        goal_coordinates = [None, None]
        children_list_global = [[]]
        cost_to_node_list = [0.0]
        iterations = 100  #600
        radius = 1.0      #0.3
        i=0
        loop_iterations = 0
        max_distance_to_goal = 0.4    #0.15
        current_distance_to_goal = np.inf#######
        goal_reached = False
        path_list = []
        while i < iterations:
            loop_iterations+=1
            candidate = [uniform(0.05, 1.95), uniform(0.05, 3.95)]
            if self.check_rrt_point(candidate) == True:
                continue
            closest_node, distance_to_closest = self.find_nearest_rrt(candidate, node_list)
            provisional_parent = closest_node
            closest_node_index = node_list.index(closest_node)
            children_list_global.append([])
            node_list.append(candidate)
            if self.check_rrt_vortex2(candidate, provisional_parent, 20) == True:
                predecessor_list.append(None)
                cost_to_node_list.append(np.inf)
            else:
                predecessor_list.append(closest_node)
                children_list_global[closest_node_index].append(candidate)
                cost_to_provisional_parent = cost_to_node_list[node_list.index(closest_node)]
                if cost_to_provisional_parent != np.inf:
                    cost_to_node_list.append(cost_to_provisional_parent + distance_to_closest)
                else:
                    cost_to_node_list.append(np.inf)
            
            neighbors = self.neighbors_rrt(radius, candidate, node_list)
            #if len(neighbors) == 0:
             #   continue
            current_parent_and_distance = [None, np.inf]
            new_parent = False
            for k, possible_parent in enumerate(neighbors):
                current_weight = cost_to_node_list[node_list.index(possible_parent)] + self.distance_rrt(possible_parent, candidate)
                if current_weight < current_parent_and_distance[1] and current_weight < cost_to_node_list[-1]:
                    #if self.check_rrt_vortex_for_collision(candidate, possible_parent) == False:
                    if self.check_rrt_vortex2(candidate, possible_parent, 20) == False:
                        current_parent_and_distance = [possible_parent, current_weight]
                        new_parent = True  
                    else:
                        continue
            #if current_parent_and_distance[0] == None:
             #   continue
            if new_parent == True:
                if predecessor_list[-1] != None:
                    children_list_global[node_list.index(predecessor_list[-1])].remove(candidate)
                predecessor_list[-1] = current_parent_and_distance[0]
                children_list_global[node_list.index(predecessor_list[-1])].append(candidate)
                cost_to_node_list[-1] = current_parent_and_distance[1]
            for to_be_improved in neighbors:
                index_of_to_be_improved = node_list.index(to_be_improved)
                new_cost = cost_to_node_list[-1] + self.distance_rrt(candidate, to_be_improved)
                if new_cost < cost_to_node_list[index_of_to_be_improved]:
                    #if self.check_rrt_vortex_for_collision(candidate, to_be_improved) == False:
                    if self.check_rrt_vortex2(candidate, to_be_improved, 20) == False:
                        delta_cost = cost_to_node_list[index_of_to_be_improved] - new_cost
                        if predecessor_list[index_of_to_be_improved] != None:
                            children_list_global[node_list.index(predecessor_list[index_of_to_be_improved])].remove(to_be_improved)
                        predecessor_list[index_of_to_be_improved] = candidate
                        children_list_global[-1].append(to_be_improved)
                        self.rrt_update_cost(to_be_improved, children_list_global[node_list.index(to_be_improved)], children_list_global, cost_to_node_list, node_list, delta_cost)
                    else:
                        continue
            candidate_distance_to_goal = self.distance_rrt(candidate, p_finish)
            if candidate_distance_to_goal < max_distance_to_goal and candidate_distance_to_goal< current_distance_to_goal\
                and self.check_rrt_vortex2(candidate, p_finish, 20) == False and cost_to_node_list[-1] != np.inf:
                #and self.check_rrt_vortex_for_collision(candidate, p_finish) == False:
                
                
                goal_reached = True
                goal_coordinates = candidate
                current_distance_to_goal = candidate_distance_to_goal
            i+=1
        current_path_length = cost_to_node_list[node_list.index(goal_coordinates)]
            
        if goal_reached == False:
            #self.get_logger().info('Goal not found')
            return 0
        node_to_be_added = goal_coordinates
        while not p_start in path_list:
            path_list.insert(0, node_to_be_added)
            node_to_be_added = predecessor_list[node_list.index(node_to_be_added)]
        xy_3d = path_list
        #self.get_logger().info('at important point')
        #elf.get_logger().info(f'p0: {p_start}, p1: {p1}, list: {xy_3d}')
        
        if len(xy_3d) == 1:
            xy_3d.append(xy_3d[0])

        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(xy_3d) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(xy_3d)
        ]
        # Replace the last point with the exac value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position
        # Now we have a waypoint path with the x and y component computed by
        # our path finding algorithm and z is a linear interpolation between
        # the z coordinate of the start and the goal pose.

        # now we need to compute our desired heading (yaw angle) while we
        # follow the waypoints. We choose a not-so-clever approach by
        # keeping the yaw angle from our start pose and only set the yaw
        # angle to the desired yaw angle from the goal pose for the very last
        # waypoint
        q0 = p0.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
        q1 = p1.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])

        # replace the very last orientation with the orientation of our
        # goal pose p1.
        #q = quaternion_from_euler(0.0, 0.0, yaw0)
        q = quaternion_from_euler(0.0, 0.0, yaw1)#yaw0
        orientations = [Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                        ] * len(points_3d)
        q = quaternion_from_euler(0.0, 0.0, yaw1)
        orientations[-1] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.get_logger().info(f'path length = {current_path_length}')
        self.get_logger().info(f'loop iterations: {loop_iterations}')

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        #self.get_logger().info('almost done')
        return [{'path': path, 'collision_indices': []}, current_path_length]

    def compute_simple_path_segment(self,
                                    p0: Pose,
                                    p1: Pose,
                                    check_collision=True):
        p0_2d = world_to_matrix(p0.position.x, p0.position.y, self.cell_size)
        p1_2d = world_to_matrix(p1.position.x, p1.position.y, self.cell_size)
        # now we should/could apply some sophisticated algorithm to compute
        # the path that brings us from p0_2d to p1_2d. For this dummy example
        # we simply go in a straight line. Not very clever, but a straight
        # line is the shortes path between two points, isn't it?
        line_points_2d = compute_discrete_line(p0_2d[0], p0_2d[1], p1_2d[0],
                                               p1_2d[1])
        if check_collision:
            collision_indices = self.has_collisions(line_points_2d)
        else:
            collision_indices = []

        # Convert back our matrix/grid_map points to world coordinates. Since
        # the grid_map does not contain information about the z-coordinate,
        # the following list of points only contains the x and y component.
        xy_3d = multiple_matrix_indeces_to_world(line_points_2d, self.cell_size)

        # it might be, that only a single grid point brings us from p0 to p1.
        # in this duplicate this point. this way it is easier to handle.
        if len(xy_3d) == 1:
            xy_3d.append(xy_3d[0])
        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(xy_3d) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(xy_3d)
        ]
        # Replace the last point with the exac value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position
        # Now we have a waypoint path with the x and y component computed by
        # our path finding algorithm and z is a linear interpolation between
        # the z coordinate of the start and the goal pose.

        # now we need to compute our desired heading (yaw angle) while we
        # follow the waypoints. We choose a not-so-clever approach by
        # keeping the yaw angle from our start pose and only set the yaw
        # angle to the desired yaw angle from the goal pose for the very last
        # waypoint
        q0 = p0.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
        q1 = p1.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])

        # replace the very last orientation with the orientation of our
        # goal pose p1.
        #q = quaternion_from_euler(0.0, 0.0, yaw0)
        q= quaternion_from_euler(0.0, 0.0, yaw1)
        orientations = [Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                        ] * len(points_3d)
        q = quaternion_from_euler(0.0, 0.0, yaw1)
        orientations[-1] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        return {'path': path, 'collision_indices': collision_indices}

    def reset_internals(self):
        self.target_viewpoint_index = -1
        self.recomputation_required = True
        self.state = State.UNSET

    def serve_start(self, request, response):
        if self.state != State.NORMAL_OPERATION:
            self.get_logger().info('Starting normal operation.')
            self.reset_internals()
        self.state = State.NORMAL_OPERATION
        response.success = True
        return response

    def serve_stop(self, request, response):
        if self.state != State.IDLE:
            self.get_logger().info('Asked to stop. Going to idle mode.')
        response.success = self.do_stop()
        return response

    def do_stop(self):
        self.state = State.IDLE
        if self.path_finished_client.call(Trigger.Request()).success:
            self.reset_internals()
            self.state = State.IDLE
            return True
        return False

    def handle_mission_completed(self):
        self.get_logger().info('Mission completed.')
        if not self.do_stop():
            self.get_logger().error(
                'All waypoints completed, but could not '
                'stop the path_follower. Trying again...',
                throttle_duration_sec=1.0)
            return
        self.state = State.IDLE

    def sort_viewpoints(self, unsorted_viewpoints: Viewpoints):
        #self.get_logger().info('bin')
        sorted_viewpoint_indices = [0]
        sorted_viewpoints = Viewpoints()
        viewpoint_list = []
        i = 0       
        viewpoint_pose_list = [v.pose for v in unsorted_viewpoints.viewpoints]
        while len(sorted_viewpoint_indices) < len(unsorted_viewpoints.viewpoints):
            next_viewpoint_index_and_distance = [None, None]
            first_new = True
            for k, current_pose in enumerate(viewpoint_pose_list):
                if k in sorted_viewpoint_indices:
                    continue
                current_distance = self.compute_viewpoint_distance(viewpoint_pose_list[sorted_viewpoint_indices[i]], current_pose)
                if first_new == True:
                    next_viewpoint_index_and_distance = [k, current_distance]
                    first_new = False
                elif current_distance < next_viewpoint_index_and_distance[1]:
                    next_viewpoint_index_and_distance = [k, current_distance]
            sorted_viewpoint_indices.append(next_viewpoint_index_and_distance[0])
            i += 1
        for l in range(len(unsorted_viewpoints.viewpoints)):
            viewpoint_list.append(unsorted_viewpoints.viewpoints[sorted_viewpoint_indices[l]])
        sorted_viewpoints.viewpoints = viewpoint_list
        return sorted_viewpoints
        
    def compute_viewpoint_distance(self, pose0: Pose, pose1:Pose):
        return np.sqrt((pose0.position.x-pose1.position.x)**2 + (pose0.position.y-pose1.position.y)**2)

    def compute_new_path(self, viewpoints: Viewpoints):
        i = self.find_first_uncompleted_viewpoint(viewpoints)
        # start position and is treated differently.
        if i < 1:
            return
        #v0 = viewpoints.viewpoints[i-1]
        #viewpoints = self.sort_viewpoints(v0, viewpoints)
        # complete them all.
        # We do nothing smart here. We keep the order in which we received
        # the waypoints and connect them by straight lines.
        # No collision avoidance.
        # We only perform a collision detection and give up in case that
        # our path of straight lines collides with anything.
        # Not very clever, eh?

        # now get the remaining uncompleted viewpoints. In general, we can
        # assume that all the following viewpoints are uncompleted, since
        # we complete them in the same order as we get them in the
        # viewpoints message. But it is not that hard to double-check it.
        viewpoint_poses = [
            v.pose for v in viewpoints.viewpoints[i:] if not v.completed
        ]
        # get the most recently visited viewpoint. Since we do not change
        # the order of the viewpoints, this will be the viewpoint right
        # before the first uncompleted viewpoint in the list, i.e. i-1
        p0 = viewpoints.viewpoints[i - 1].pose
        viewpoint_poses.insert(0, p0)
        #self.get_logger().info(f'viewpointlist unsorted: {len(viewpoint_poses)}')
        #viewpoint_poses.pop(0)
        #viewpoint_poses = self.sort_viewpoints(p0, viewpoint_poses)
        #self.get_logger().info(f'sorted: {len(viewpoint_poses)}')
        # now we can finally call our super smart function to compute
        # the path piecewise between the viewpoints
        path_segments = []
        for i in range(1, len(viewpoint_poses)):
            #segment = self.compute_simple_path_segment(viewpoint_poses[i - 1],
            #                                           viewpoint_poses[i])

            # alternatively call your own implementation
            #segment = self.compute_a_star_segment(viewpoint_poses[i - 1],
             #                                     viewpoint_poses[i])
            segment = self.compute_rrt_star_segment22(viewpoint_poses[i-1],
                                                    viewpoint_poses[i])[0]
            path_segments.append(segment)
            #self.get_logger().info(f'seg {i}: {segment}')
        #self.get_logger().info(f'len path segments: {len(path_segments)}')
        return path_segments

    def handle_no_collision_free_path(self):
        self.get_logger().fatal('We have a collision in our current segment!'
                                'Giving up...')
        if self.do_stop():
            self.state = State.IDLE
        else:
            self.state = State.UNSET

    def do_normal_operation(self, viewpoints: Viewpoints):
        # what we need to do:
        # - check if the viewpoints changed, if so, recalculate the path
        i = self.find_first_uncompleted_viewpoint(viewpoints)
        # we completed our mission!
        if i < 0:
            self.handle_mission_completed()
            return

        if (not self.recomputation_required
            ) or self.target_viewpoint_index == i:
            # we are still chasing the same viewpoint. Nothing to do.
            return
        self.get_logger().info('Computing new path segments')
        self.target_viewpoint_index = i
        if i == 0:
            p = viewpoints.viewpoints[0].pose
            if not self.move_to_start(p, p):
                self.get_logger().fatal(
                    'Could not move to first viewpoint. Giving up...')
                if self.do_stop():
                    self.state = State.IDLE
                else:
                    self.state = State.UNSET
            return

        path_segments = self.compute_new_path(viewpoints)
        if not path_segments:
            self.get_logger().error(
                'This is a logic error. The cases that would have lead to '
                'no valid path_segments should have been handled before')
            return
        if path_segments[0]['collision_indices']:
            self.handle_no_collision_free_path()
            return
        self.set_new_path(path_segments[0]['path'])
        return

    def on_viewpoints(self, msg: Viewpoints):
        if self.state == State.IDLE:
            return
        if self.state == State.UNSET:
            if self.do_stop():
                self.state = State.IDLE
            else:
                self.get_logger().error('Failed to stop.')
            return
        if self.state == State.MOVE_TO_START:
            # nothing to be done here. We already did the setup when the
            # corresponding service was called
            return
        if self.state == State.NORMAL_OPERATION:
            #self.do_normal_operation(msg)
            
            sorted_viewpoints = self.sort_viewpoints(msg)
            self.do_normal_operation(sorted_viewpoints)


    def find_first_uncompleted_viewpoint(self, viewpoints: Viewpoints):
        for i, viewpoint in enumerate(viewpoints.viewpoints):
            if not viewpoint.completed:
                return i
        # This should not happen!
        return -1

    def on_occupancy_grid(self, msg: OccupancyGrid):
        self.occupancy_grid = msg
        self.occupancy_matrix = occupancy_grid_to_matrix(self.occupancy_grid)
        if msg.info.resolution != self.cell_size:
            self.get_logger().info('Cell size changed. Recomputation required.')
            self.recomputation_required = True
            self.cell_size = msg.info.resolution

    def init_path_marker(self):
        msg = Marker()
        msg.action = Marker.ADD
        msg.ns = 'path'
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.header.frame_id = 'map'
        msg.color.a = 1.0
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.scale.x = 0.02
        msg.scale.y = 0.02
        msg.scale.z = 0.02
        self.path_marker = msg

    def set_new_path(self, path):
        request = SetPath.Request()
        if not path:
            return False
        request.path = path
        self.set_new_path_future = self.set_path_client.call_async(request)
        return True

    def publish_path_marker(self, segments):
        msg = self.path_marker
        world_points = self.segments_to_world_points(segments)
        msg.points = [Point(x=p[0], y=p[1], z=-0.5) for p in world_points]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.path_marker_pub.publish(msg)


def main():
    rclpy.init()
    node = PathPlanner()
    exec = MultiThreadedExecutor()
    exec.add_node(node)
    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
