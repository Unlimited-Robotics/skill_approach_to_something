# Ra-Ya Skill - Approach to Something

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Unlimited-Robotics/skill_approach_to_something/graphs/commit-activity)

## Description

This skill approaches to something, it can be a tag, a face, a person, etc. it depends on the predictor that is set, it needs to be localized on a map.

Details about the logic can be found in the [Ra-Ya documentation](https://www.notion.so/Approach-to-something-83d61d4d64fe4ad19141749879a880af).

## Requirements

* [Ra-Ya controllers]: MotionController, CVController, NavigationController

## Installation

``` bash
rayasdk skills install approach_to_something
```

## Usage

This example will approach to a face using the camera nav_bottom, it will approach in a angle of 90 degrees taking as a reference the map, when the robot is at 1 m from the face it will stop.


``` python
from raya.application_base import RayaApplicationBase
from raya.skills import RayaSkillHandler

from skills.approach_to_something import SkillApproachToSomething
class RayaApplication(RayaApplicationBase):

    async def setup(self):
        self.skill_apr2something:RayaSkillHandler = \
                self.register_skill(SkillApproachToSomething)
        
        await self.skill_apr2something.execute_setup(
                setup_args={
                        'working_camera': ['nav_bottom'],
                        'predictor': 'yunet_face',
                    },
            )


    async def main(self):
        execute_result = await self.skill_apr2something.execute_main(
                execute_args={
                        'angle_to_goal': 90,
                        'distance_to_goal': 1.0,
                    },
                callback_feedback=self.cb_feedback
            )
        self.log.debug(execute_result)


    async def finish(self):
        await self.skill_apr2something.execute_finish()


    async def cb_feedback(self, feedback):
        self.log.debug(feedback)

```

## Exceptions

| Exception | Value (error_code, error_msg) |
| :-------  | :--- |
| ERROR_INVALID_ANGLE | (1, 'Invalid angle, must be between -180 and 180')
| ERROR_INVALID_PREDICTOR | (2, 'Invalid predictor') |
| ERROR_IDENTIFIER_NOT_DEFINED | (3, 'Identifier must be defined') |
| ERROR_NOT_LOCALIZED | (4, f'The robot must be localized') |
| ERROR_INITIAL_ANGLE_TOO_FAR | (5, Custom message) |
| ERROR_NO_TARGET_FOUND | (6, 'Not target found after {NO_TARGET_TIMEOUT_LONG}') |
| ERROR_TOO_DISALIGNED | (7, Custom message) |
| ERROR_TOO_CLOSE_TO_TARGET | (8, Custom message) |
| ERROR_TOO_FAR_TO_TARGET | (9, Custom message) |
| ERROR_MOVING | (10, Custom message) |
| ERROR_NAVIGATION | (11, Custom message) |

## Arguments

### Setup

#### Required

| Name              | Type     | Description |
| :--------------- | :------: | :---- |
| working_cameras   | [string] | List of cameras to use. Take into account that more cameras means more models to activate. |
| predictor         | string    | Cv model to use, Ex: `yunet_face` |

#### Default

| Name          | Type | Default value | Description |
| :---------------- | :------: | :------: | :---- |
| fsm_log_transitions | boolean | True | Shows the log of the transitions of the fsm. |
| identifier | string | '' | Some models need a identifier to get an specific prediction.  |

### Execute

### Required

None

#### Default

| Name                             | Type    | Default value    | Description |
| :--------------------------------|:-------:| :------------:| :-----------|
| distance_to_goal                 | float   | 1.0   |  Distance to the goal, in meters; the robot stops upon reaching this distance to the goal and corrects the angle. |
| angle_to_goal                    | float   | 0.0   | Angle in degrees; approximation angle to target. |
| intersection_threshold           | float   | 0.2   | Value to set intersection zone.            |
| angular_velocity                 | int     | 10    | Angular velocity. |
| linear_velocity                  | float   | 0.1   | Linear velocity.  |
| min_correction_distance          | float   | 0.5   | Minimum correction distance.            |
| save_trajectory                  | boolean | False | Saves the skill data for analysis, it being stored on the dat folder of the app.            |
| step_size                        | float   | 0.2   | Step size.            |
| predictions_to_average           | int     | 6     | Number of predictions to average.            |
| max_x_error_allowed              | float   | 0.02  | Maximum allowed error in the x-axis.            |
| max_y_error_allowed              | float   | 0.1   | Maximum allowed error in the y-axis.            |
| max_angle_error_allowed          | float   | 5.0   | Maximum allowed error in angle.            |
| predictions_timeout              | float   | 0     | Timeout to abort fsm when tracking target is higger than tracking_threshold.            |
| tracking_threshold               | float   | 0.3   | Value to check if the distance between the position of the current prediction and the position of the previous prediction is greater than the tracking threshold.            |
| position_state_threshold         | float   | 0.5   | Value to check if the distance between the first prediction (P0) and the current prediction is greater than the established in position_state_threshold.            |
| max_allowed_distance             | float   | 2.0   | Maximum allowed distance from target.             |
| allowed_motion_tries             | int     | 10    | Maximum allowed correction tries. |
| allow_previous_predictions       | boolean | True  | It takes into account the previous prediction.            |
| y_offset                         | float   | 0.0   | Y-axis offset.              |
