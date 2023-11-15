from ast import literal_eval
from raya.application_base import RayaApplicationBase
from raya.tools.image import show_image, draw_on_image
from raya.skills import RayaSkill, RayaSkillHandler

from skills.approach_to_something import SkillApproachToSomething


class RayaApplication(RayaApplicationBase):

    async def setup(self):
        self.skill_apr2something:RayaSkillHandler = \
                self.register_skill(SkillApproachToSomething)
        
        await self.skill_apr2something.execute_setup(
                setup_args={
                        'working_camera': self.camera,
                        'identifier':self.identifier,
                        'predictor': self.predictor,
                    },
            )


    async def main(self):
        execute_result = await self.skill_apr2something.execute_main(
                execute_args={
                        'angle_to_goal':self.angle_to_target,
                        'distance_to_goal': self.target_distance,
                        'angular_velocity': self.vel_ang,
                        'linear_velocity': self.vel_lin,
                        'step_size': self.step_size,
                        'max_x_error_allowed': self.max_x_err,
                        'max_y_error_allowed': self.max_y_err,
                        'max_angle_error_allowed': self.max_a_err,
                        'max_allowed_distance': self.max_distance,
                        'save_trajectory': self.save_trajectory,
                        'y_offset': self.y_offset
                    },
                callback_feedback=self.cb_feedback
            )
        self.log.debug(execute_result)


    async def finish(self):
        await self.skill_apr2something.execute_finish()


    async def cb_feedback(self, feedback):
        self.log.debug(feedback)


    def get_arguments(self):
        self.camera = self.get_argument('-c', '--camera', 
                type=str, 
                required=True,
                help='name of camera to use'
            )   
        self.angle_to_target = self.get_argument('-a', '--angle', 
                type=float, 
                required=True,
                help='Angle to approach'
            )  
        self.target_distance = self.get_argument('-d', '--distance-to-target', 
                type=float, 
                required=False,
                default=1.0,
                help='Final target distance'
            )  
        self.predictor = self.get_argument('-p', '--predictor', 
                type=str, 
                required=True,
                help='predictor to use'
            )   
        self.identifier = self.get_argument('-i', '--identifier', 
                type=str,
                default='',
                help='identifier to be used'
            )  
        self.save_trajectory = self.get_flag_argument('--save-trajectory',
                help='Enable saving trajectory',
            )
        self.step_size = self.get_argument('--step-size',
                type=float,
                help='size of last steps movements',
                default=0.2,
            )
        self.max_x_err = self.get_argument('--max-x-err',
                type=float,
                help='max error in x',
                default=0.02,
            )
        self.max_y_err = self.get_argument('--max-y-err',
                type=float,
                help='max error in y',
                default=0.05,
            )
        self.max_a_err = self.get_argument('--max-a-err',
                type=float,
                help='max angle error',
                default=5.0,
            )
        self.vel_ang = self.get_argument('--vel-ang',
                type=float,
                help='Angular velocity',
                default=10.0,
            )
        self.vel_lin = self.get_argument('--vel-lin',
                type=float,
                help='linerar velocity',
                default=0.1,
            )
        self.max_distance = self.get_argument(
            '--max-distance',
            type= float,
            help='maximum distance allowed to start approaching',
            default=2.0)
        self.y_offset = self.get_argument(
            '--y-offset',
            type= float,
            help='Offset in y axis',
            default=0.0)        
        try:
            self.identifier = literal_eval(self.identifier)
        except:
            pass
