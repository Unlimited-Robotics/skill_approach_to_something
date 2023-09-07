import math
import queue
import time
import os
import json
import queue
import numpy as np
import asyncio
from raya.exceptions import *
from raya.tools.filesystem import open_file, check_file_exists
from raya.utils.internal_filesystem import assert_file_exists
from datetime import datetime
import time

from raya.enumerations import POSITION_UNIT, ANGLE_UNIT
import tf_transformations as tftr
from transforms3d import euler

from raya.controllers import MotionController
from raya.controllers import CVController, NavigationController
from raya.skills import RayaFSMSkill

from .constants import *
import threading



class SkillApproachToSomething(RayaFSMSkill):

    ### FSM ###

    STATES = [
            'READ_TARGET',
            'GO_TO_INTERSECTION',
            'READ_TARGET_2',
            'ROTATE_TO_TARGET',
            'STEP_N',
            'READ_TARGET_N',
            'CENTER_TO_TARGET',
            'READ_TARGET_FINAL',
            'MOVE_LINEAR_FINAL',
            'END'
        ]

    INITIAL_STATE = 'READ_TARGET'

    END_STATES = [
        'END',
    ]

    STATES_TIMEOUTS = {
        'READ_TARGET' : (NO_TARGET_TIMEOUT_LONG, ERROR_NO_TARGET_FOUND),
        # 'READ_TARGET_N' : (NO_TARGET_TIMEOUT_LONG, ERROR_NO_TARGET_FOUND),
        # 'READ_TARGET_FINAL' : (NO_TARGET_TIMEOUT_LONG, ERROR_NO_TARGET_FOUND),
    }


    ### SKILL ###

    REQUIRED_SETUP_ARGS = [
            'working_camera'
        ]

    DEFAULT_SETUP_ARGS = {
            'log_transitions':True,
            'identifier': '',
        }

    REQUIRED_EXECUTE_ARGS = [
            'predictor'
        ]

    DEFAULT_EXECUTE_ARGS = {
            'distance_to_goal': 1.0,
            'angle_to_goal': 0.0,
            'intersection_threshold': 0.2,
            'angular_velocity': 10,
            'linear_velocity': 0.1,
            'min_correction_distance': MIN_CORRECTION_DISTANCE,
            'save_trajectory': False,
            'step_size': 0.2,
            'predictions_to_average': 6,
            'max_x_error_allowed': 0.02,
            'max_y_error_allowed': 0.1,
            'max_angle_error_allowed': 5.0,
            'predictions_timeout': 0,
            'tracking_threshold': 0.3,
            'position_state_threshold': 0.5,
        }

    ### SKILL METHODS ###

    async def setup(self):
        self.timer1 = None
        self.task_exception = None
        self.step_task = None

        self.motion:MotionController = await self.get_controller('motion')
        self.navigation:NavigationController = await self.get_controller('navigation')

        self.cv:CVController = await self.get_controller('cv')

        self.mutex_cb_detections= threading.Lock()
        # TODO: Definir los params dependiendo del modelo
        self.predictor_handler = await self.cv.enable_model(
                name=self.execute_args['predictor'], 
                source=self.setup_args['working_camera'],
                model_params = {'depth':True}
            )

    async def finish(self):
        # Disable models
        pass


    ### HELPERS ###


    def setup_variables(self):
        self.handler_name = type(self.predictor_handler).__name__
        self.approach = HANDLER_NAMES[self.handler_name]
        if self.execute_args['angle_to_goal'] > 0:
            self.target_angle = self.execute_args['angle_to_goal'] - 180
        else:
            self.target_angle = self.execute_args['angle_to_goal'] + 180
        self.p_prediction = []
        self.initial_pos = []
        self.previous_goal = []
        self.__timestamps = []
        self.__face_positions = []
        self.__robot_positions = []
        self.__previous_timestamp = None
        self.__commands = []
        self.command = ()
        self.tick = False
        self.near_prediction = False, 
        self.closest_prediction = None

        #flags
        self.is_there_detection = False
        self.waiting_detection = True
        self.wait_until_complete_queue = True
        self.is_final_step = False

        #calculations
        self.correct_detection = None
        self.angle_intersection_goal = None
        self.angle_robot_intersection = None
        self.angular_sign = None
        self.angle_robot_goal = None
        self.linear_distance = None

        self.__predictions_queue= queue.Queue()

        self.additional_distance= self.execute_args['min_correction_distance']

        self.predictor_handler.set_detections_callback(
                callback=self._callback_predictions,
                as_dict=True,
                call_without_detections=True
            )

    def validate_arguments(self):
        # TODO: Validar que el predictor exista
        self.setup_variables()
        if self.execute_args['angle_to_goal'] > 180.0 or \
                self.execute_args['angle_to_goal'] < -180.0:
            self.abort(*ERROR_INVALID_ANGLE)
        if not self.handler_name in HANDLER_NAMES:
            self.abort(*ERROR_INVALID_PREDICTOR)
        if not self.setup_args['identifier'] and \
                HANDLER_NAMES[self.handler_name] is not None:
            self.abort(*ERROR_IDENTIFIER_NOT_DEFINED)


    async def rotate_and_move_linear(self):
        direction = 'left' if np.sign(self.angle_robot_intersection) > 0 else 'right'
        self.log.debug(
                f'Rotating {direction} {self.angle_robot_intersection}'
            )
        try:
            if abs(self.projected_error_y) > self.execute_args['max_y_error_allowed']:
                self.log.debug(f'rotating because projected error exceed limit {self.projected_error_y}')
                await self.motion.rotate(
                        angle=abs(self.angle_robot_intersection), 
                        angular_speed=self.execute_args['angular_velocity'] * np.sign(self.angle_robot_intersection), 
                        wait=True
                    )
            await self.motion.move_linear(
                    distance=self.linear_distance, 
                    x_velocity=self.execute_args['linear_velocity'], 
                    wait=True
                )
        except Exception as e:
            self.task_exception = e
    

    def start_detections(self, wait_complete_queue=True):
        self.is_there_detection = False
        self.waiting_detection = True
        self.wait_until_complete_queue = wait_complete_queue
        self.correct_detection = None


    def stop_detections(self):
        self.waiting_detection = False


    async def validate_initial_condition(self):
        if not await self.navigation.is_localized():
            self.abort(*ERROR_NOT_LOCALIZED)
        _, _, current_robot_angle = await self.navigation.get_position()
        angle_error = abs(self.angle_difference(
                current_robot_angle, self.execute_args['angle_to_goal']
            ))
        if angle_error > MAX_INITIAL_ANGLE_ERROR:
            self.abort(
                    ERROR_INITIAL_ANGLE_TOO_FAR,
                    f'The initial angle ({current_robot_angle:.2f}) is too far'
                    f' from the target angle ({self.execute_args["angle_to_goal"]:.2f})'
                )
            
    
    async def check_initial_position(self):
        x_final = False
        y_final = False
        robot_position = await self.navigation.get_position(
                pos_unit=POSITION_UNIT.METERS, 
                ang_unit=ANGLE_UNIT.DEGREES
            )
        goal = self.correct_detection
        self.initial_pos = self.correct_detection
        self.closest_prediction = self.correct_detection
        distance_x, distance_y = self.__get_relative_coords(
                goal[:2], 
                robot_position[:2], 
                self.target_angle
            )
        distance_x = abs(distance_x) - self.execute_args['distance_to_goal']
        if abs(distance_x) <= self.execute_args['max_x_error_allowed']:
            x_final = True              
        if abs(distance_y) <= self.execute_args['max_y_error_allowed']:
            y_final = True
        if x_final == True and y_final == True:  
            return
        ini_target_distance = self.get_euclidean_distance(
                robot_position[:2], 
                goal
            )
        if ini_target_distance < (self.execute_args['distance_to_goal'] + 
                                  self.execute_args['min_correction_distance']):
            self.abort(
                    ERROR_TOO_CLOSE_TO_TARGET,
                    f'Robot is too close to the target. It is '
                    f'{ini_target_distance:.2f}, and it must be at least the '
                    f'distance to goal ({self.execute_args["distance_to_goal"]:.2f}) '
                    f'+ MIN_CORRECTION_DISTANCE ({MIN_CORRECTION_DISTANCE})'
                )


    async def planning_calculations(self):
        if not self.correct_detection:
            self.correct_detection = self.previous_goal
        self.p_prediction = [
                self.correct_detection[0], 
                self.correct_detection[1], 
                self.target_angle
            ]
        self.log.warn(
                f'self.initial_pos: {self.p_prediction}'
            )
        int_info, self.angle_robot_intersection, \
        self.distance_to_inter, self.distance = await self.get_intersection_info(self.p_prediction)     
        self.intersection, self.before = int_info
        # self.log.warn(
        #         f'{self.intersection}: {self.before} '
        #         f'{self.distance} {self.distance_to_inter} {self.angle_robot_intersection}'
        #     ) 
        robot_position = await self.navigation.get_position(
            pos_unit=POSITION_UNIT.METERS, 
            ang_unit=ANGLE_UNIT.DEGREES
        )         
        distance_x, self.projected_error_y = self.__get_relative_coords(
                self.correct_detection[:2], 
                robot_position[:2], 
                self.target_angle
            )
        self.projected_error_x = abs(distance_x) - self.execute_args['distance_to_goal']
        self.log.debug(
                f'distance to final point {distance_x} {self.projected_error_y}'
            )
        self.angle_robot_goal = self.get_angle(robot_position, self.correct_detection)


    async def record_state_info(self):
        while True:
            if self.tick:
                self.tick=False
            if not self.initial_pos:
                _, goal = await self.__update_predictions(
                        buffer=5,
                        near_prediction=True
                    )   
            else:     
                _, goal = await self.__update_predictions(
                        buffer=5,
                        closest_prediction=self.initial_pos
                    )   
            robot_position = await self.navigation.get_position(
                    pos_unit=POSITION_UNIT.METERS, 
                    ang_unit=ANGLE_UNIT.DEGREES
                )
            self.__timestamps.append(time.time())
            if goal is None:
                self.__face_positions.append([])
            else:
                self.__face_positions.append(list(goal))
            self.__robot_positions.append(list(robot_position))
            self.__commands.append(self.command)
            self.record_dict = {
                    'params': vars(self.execute_args), 
                    'time': self.__timestamps, 
                    'robot_position': self.__robot_positions, 
                    'face_position': self.__face_positions, 
                    'cmd': self.__commands
                }
            self.record_dict['params'].pop('predictor',None)
            now_time = datetime.now()

            wished_format = "%Y%m%d_%H%M%S"
            date = now_time.strftime(wished_format)
            if self.__previous_timestamp:
                previous_file = f"dat:trajectory_{self.__previous_timestamp}.json"
                previous_file_path = assert_file_exists(previous_file)
                os.remove(previous_file_path)  
            self.__previous_timestamp = date
            with open_file(f"dat:trajectory_{date}.json", "w") as archivo_json:
                json.dump(self.record_dict, archivo_json)
            await self.sleep(0.005)
        
        

    async def get_intersection_info(self, p_prediction):
        line = [None, None, None]
        robot_position = await self.navigation.get_position(
                pos_unit=POSITION_UNIT.METERS, 
                ang_unit=ANGLE_UNIT.DEGREES
            )
        line[0] = [p_prediction[0], p_prediction[1], self.target_angle]
        distance = self.get_euclidean_distance(
                robot_position[:2], 
                p_prediction[:2]
            )
        line[1] = self.get_proyected_point(
                p_prediction[0], 
                p_prediction[1], 
                self.target_angle, 
                distance+0.5
            )
        line[2] = self.get_proyected_point(
                p_prediction[0], 
                p_prediction[1], 
                self.target_angle, 
                self.execute_args['distance_to_goal']+0.5
            )
        goal_point = self.get_proyected_point(
                p_prediction[0], 
                p_prediction[1], 
                self.target_angle, 
                self.execute_args['distance_to_goal']
            )
        linear_distance = self.get_euclidean_distance(
                robot_position, 
                line[2]
            )
        angle_robot_intersection = self.get_angle(robot_position, goal_point)
        distance_to_inter = self.__get_point_line_distance(
                robot_position[:2], 
                [line[0][:2], line[1]]
            )
        return self.__get_intersection(
                line[0], 
                line[1],
                line[2], 
                robot_position
            ), angle_robot_intersection, distance_to_inter, linear_distance
    
    
    
    def _callback_predictions(self, predictions, timestamp):
        if predictions:
            __predictions = predictions
            __predictions['timestamp'] = timestamp[0]+timestamp[1]/1e9
            if self.waiting_detection:
                self.__predictions_queue.put(predictions)
                if self.__predictions_queue._qsize() == \
                        self.execute_args['predictions_to_average'] or \
                        not self.wait_until_complete_queue:                
                    self.__update_predictions()
                

    def __update_predictions(self):
        predicts = []
        temporal_queue = queue.Queue()

        while not self.__predictions_queue.empty():
            prediction = self.__predictions_queue.get()
            goal = self.__proccess_prediction(prediction)
            if not goal:
                continue
            temporal_queue.put(prediction)
            predicts.append(goal)

        if (len(predicts) == self.execute_args['predictions_to_average'] or 
            not self.wait_until_complete_queue):
            correct_detection = self.__process_multiple_detections(predicts)
            if correct_detection:
                self.correct_detection = correct_detection
                self.is_there_detection = True
                self.waiting_detection = False
                return
            else:
                temporal_queue.get() # discarding last value 

        while not temporal_queue.empty():
            self.__predictions_queue.put(temporal_queue.get())


    def __proccess_prediction(self, prediction):
        goal = None
        frames_procesed = 0
        possible_goals = []
        for pred in prediction.values():
            if type(pred) == float:
                continue
            if not self.setup_args['identifier'] and \
                    self.approach is not None:
                if self.setup_args['identifier'] != pred[self.approach]:
                    continue
            if pred['center_point_map']:
                goal = pred['center_point_map'][:2]
                proximity = None
                if self.closest_prediction:
                    proximity = self.get_euclidean_distance(
                            goal, 
                            self.closest_prediction
                        )
                    if proximity <= self.execute_args['position_state_threshold']:
                        possible_goals.append(
                                (
                                    goal, 
                                    pred['distance'], 
                                    proximity
                                )
                            )
        if possible_goals:
            if self.near_prediction:
                possible_goals = sorted(
                        possible_goals, 
                        key=lambda x: x[1]
                    )
            else:
                possible_goals = sorted(
                        possible_goals, 
                        key=lambda x: x[2]
                    )
            frames_procesed += 1
            goal = possible_goals[0][0]
        if goal:
            self.previous_goal = goal
        return goal


    def __process_multiple_detections(self, predictions):
        # Step 1: Calculate the mean of the list of predictions (arrays of three values)
        predictions_np = np.array(predictions)
        valid_predictions = predictions_np[~np.isnan(
            predictions_np).any(axis=1)]

        if len(valid_predictions) == 0:
            return None  # Return None if all positions have NaN values

        # Step 2: Calculate the mean of the valid predictions (arrays of three values)
        mean_prediction = np.mean(valid_predictions, axis=0)

        # Step 3: Get the values below the mean
        below_mean_values = valid_predictions[
            (abs(valid_predictions-mean_prediction)<=0.3).sum(axis=1)>1, :]
        below_mean_values = below_mean_values[
            (abs(below_mean_values[:,-1]-mean_prediction[-1])<=10), :]
        if len(below_mean_values) == 0:
            return None
        # Step 5: Calculate the mean of the values below the mean
        mean_below_mean = np.mean(below_mean_values, axis=0)

        return mean_below_mean.tolist()
    

    async def calculate_intersection_info(self):
        self.p_prediction = [
                self.correct_detection[0], 
                self.correct_detection[1], 
                self.target_angle
            ]
        self.log.warn(
                f'self.initial_pos: {self.p_prediction}'
            )
        int_info, self.angle_robot_intersection, \
        self.distance_to_inter, self.distance = await self.get_intersection_info(self.p_prediction)     
        self.intersection, self.before = int_info
        # self.log.warn(
        #         f'{self.intersection}: {self.before} '
        #         f'{self.distance} {self.distance_to_inter} {self.angle_robot_intersection}'
        #     ) 


    def __get_point_line_distance(self, punto, linea):
        x, y = punto
        x1, y1 = linea[0]
        x2, y2 = linea[1]
        longitud_linea = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if longitud_linea == 0:
            return 0
        t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (longitud_linea**2)
        proyeccion_x = x1 + t * (x2 - x1)
        proyeccion_y = y1 + t * (y2 - y1)
        distancia = math.sqrt((x - proyeccion_x)**2 + (y - proyeccion_y)**2)
        return distancia


    def __get_relative_coords(self, punto_a, punto_b, angulo_direccion):
        x_a, y_a = punto_a
        x_b, y_b = punto_b
        delta_x = x_b - x_a
        delta_y = y_b - y_a
        angulo_rad = math.radians(angulo_direccion)
        x_rel = delta_x * math.cos(angulo_rad) + delta_y * math.sin(angulo_rad)
        y_rel = delta_y * math.cos(angulo_rad) - delta_x * math.sin(angulo_rad)
        return x_rel, y_rel
    

    def get_angle(self, robot_point, line_point1):
        robot_angle_rad = np.radians(robot_point[2])
        robot_direction = np.array(
                [np.cos(robot_angle_rad), np.sin(robot_angle_rad)]
            )
        line_direction = np.array(line_point1[:2]) - np.array(robot_point[:2])
        line_direction /= np.linalg.norm(line_direction)  
        angle_rad = np.arctan2(
                line_direction[1], 
                line_direction[0]) - np.arctan2(
                                            robot_direction[1], 
                                            robot_direction[0]
                                        )
        angle_deg = np.degrees(angle_rad)
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg < -180:
            angle_deg += 360
        return angle_deg
    

    def __get_intersection(self, line_point1, line_point2, line_point3, robot_point):
        p1 = np.array(line_point1[:2])
        p2 = np.array(line_point2)
        p0 = np.array(robot_point[:2])
        before = False
        intersection = None
        line_direction = p2 - p1
        p1_to_robot = p0 - p1
        robot_angle_rad = np.radians(robot_point[2])        
        robot_direction = np.array(
                [np.cos(robot_angle_rad), np.sin(robot_angle_rad)]
            )
        perpendicular_direction = np.array(
                [-robot_direction[1], robot_direction[0]]
            )  
        dot_product_line_perpendicular = np.dot(
                line_direction, 
                perpendicular_direction
            )
        if dot_product_line_perpendicular == 0:
            return intersection, before
        t = np.dot(
                p1_to_robot, perpendicular_direction
            ) / dot_product_line_perpendicular
        intersection = p1 + t * line_direction
        intersection_direction = np.dot(line_direction, intersection - p1)
        if intersection_direction >= 0:
            before = True
        robot_to_intersection = intersection - p0
        dot_product_robot_intersection = np.dot(
                robot_direction, 
                robot_to_intersection
            )
        if dot_product_robot_intersection < 0:
            return intersection, before
        return intersection, before


    def get_proyected_point(self, x, y, angle, distance):
        angulo_rad = math.radians(angle)
        x_final = x + distance * math.cos(angulo_rad)
        y_final = y + distance * math.sin(angulo_rad)
        return (x_final, y_final)
    

    def get_angle(self, robot_point, target_point):
        dx = target_point[0] - robot_point[0]
        dy = target_point[1] - robot_point[1]
        angulo_radianes = math.atan2(dy, dx) - math.radians(robot_point[2])
        angulo_grados = math.degrees(angulo_radianes)
        return angulo_grados
    

    # TOOLS


    def get_euclidean_distance(self, pt1, pt2):
        distance = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        return distance


    def angle_difference(self, angle_0 = -160, angle_1 = 170):
        diff = angle_1 - angle_0
        if diff>180.0: diff -= 360.0
        if diff<-180.0: diff += 360.0
        return diff
    

    def motion_running(self):
        return self.motion.is_moving()


    ### ACTIONS ###


    async def enter_READ_TARGET(self):
        self.validate_arguments()
        self.start_detections()
        await self.validate_initial_condition()
        if self.execute_args['save_trajectory']:
            self.step_task = asyncio.create_task(self.record_state_info())


    async def enter_GO_TO_INTERSECTION(self):
        await self.planning_calculations()
        self.linear_distance = self.distance
        if abs(self.distance_to_inter) > MAX_MISALIGNMENT:
            self.abort(
                    ERROR_TOO_DISALIGNED,
                    'The robot is disaligned by '
                    f'{abs(self.distance_to_inter)} meters, max '
                    f'{MAX_MISALIGNMENT} is allowed.'
                )
        self.step_task = asyncio.create_task(self.rotate_and_move_linear())


    async def enter_READ_TARGET_2(self):
        self.start_detections(wait_complete_queue=False)
        self.timer1 = time.time()
        

    async def enter_ROTATE_TO_TARGET(self):
        robot_position = await self.navigation.get_position(
                pos_unit=POSITION_UNIT.METERS, 
                ang_unit=ANGLE_UNIT.DEGREES
            )
        angle_to_rotate = self.get_angle(
                robot_position, 
                self.initial_pos
            )
        direction = 'left' if angle_to_rotate > 0 else 'right'
        self.log.debug(
                f'Rotating {direction} {abs(angle_to_rotate)}'
            )
        await self.motion.rotate(
                angle=abs(angle_to_rotate), 
                angular_speed=\
                    self.execute_args['angular_velocity'] * np.sign(angle_to_rotate), 
                wait=False
            )
        self.command = (0.0, self.execute_args['angular_velocity'] * np.sign(angle_to_rotate)) 

    
    async def enter_READ_TARGET_N(self):
        self.start_detections()
        

    async def enter_STEP_N(self):
        # self.additional_distance = 0.0
        await self.planning_calculations()
        self.linear_distance = self.execute_args['step_size']
        if self.projected_error_x<=self.execute_args['step_size']:
            self.is_final_step=True
            self.linear_distance=self.projected_error_x
        self.log.debug(f"distance: {self.projected_error_x} ")
        self.step_task = asyncio.create_task(self.rotate_and_move_linear())


    async def enter_CENTER_TO_TARGET(self):
        await self.planning_calculations()
        if abs(self.angle_robot_goal) > \
            self.execute_args['max_angle_error_allowed']:
            ang_vel=(self.execute_args['angular_velocity'] *
                    np.sign(self.angle_robot_goal))
            await self.motion.rotate(
                    angle=abs(self.angle_robot_goal), 
                    angular_speed=ang_vel, 
                    wait=False
                )
    

    async def enter_READ_TARGET_FINAL(self):
        self.start_detections()


    async def enter_MOVE_LINEAR_FINAL(self):
        await self.planning_calculations()
        linear_distance = self.projected_error_x
        self.log.debug(f"linear distance to correct {linear_distance}"
                           f" error y {self.projected_error_y}")
        if abs(linear_distance) > self.execute_args['max_x_error_allowed']:
            await self.motion.move_linear(distance=abs(linear_distance), 
                                            x_velocity=(
            self.execute_args['linear_velocity']*np.sign(linear_distance)), 
                                            wait=False)



    ### TRANSITIONS ###


    async def transition_from_READ_TARGET(self):
        if self.is_there_detection:
            if await self.check_initial_position():
                self.set_state('CENTER_TO_TARGET')
            else:
                self.set_state('GO_TO_INTERSECTION')


    async def transition_from_GO_TO_INTERSECTION(self):
        if self.step_task.done():
            await self.step_task
            if self.task_exception is not None:
                raise self.task_exception            
            self.set_state('READ_TARGET_2')


    async def transition_from_READ_TARGET_2(self):
        if (time.time()-self.timer1) > NO_TARGET_TIMEOUT_SHORT:
            self.stop_detections()
            if self.is_there_detection:
                self.set_state('READ_TARGET_N')
            else:
                self.set_state('ROTATE_TO_TARGET')


    async def transition_from_ROTATE_TO_TARGET(self):
        if not self.motion_running():
            self.motion.check_last_motion_exception()
            self.set_state('READ_TARGET_N')
    

    async def transition_from_STEP_N(self):
        if self.step_task.done():
            await self.step_task
            if self.task_exception is not None:
                raise self.task_exception
            self.set_state('READ_TARGET_N')


    async def transition_from_READ_TARGET_N(self):
        if self.is_there_detection or self.previous_goal:
            if self.is_final_step:
                self.set_state('CENTER_TO_TARGET')
            else:
                self.set_state('STEP_N')
                

    async def transition_from_CENTER_TO_TARGET(self):
        if not self.motion_running():
            self.motion.check_last_motion_exception()
            self.set_state('READ_TARGET_FINAL')


    async def transition_from_READ_TARGET_FINAL(self):
        if self.is_there_detection or self.previous_goal: 
            self.set_state('MOVE_LINEAR_FINAL')


    async def transition_from_MOVE_LINEAR_FINAL(self):
        if not self.motion_running():
            self.motion.check_last_motion_exception()
            self.set_state('END')
