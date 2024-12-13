from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

'''MATERIALS'''
bread = CustomMaterial(
    texture="Bread",
    tex_name="bread",
    mat_name="MatBread",
    tex_attrib={"type": "cube"},
    mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
)

darkwood = CustomMaterial(
    texture="WoodDark",
    tex_name="darkwood",
    mat_name="MatDarkWood",
    tex_attrib={"type": "cube"},
    mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
)

lightwood = CustomMaterial(
    texture="WoodLight",
    tex_name="lightwood",
    mat_name="MatLightWood",
    tex_attrib={"type": "cube"},
    mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
)

metal = CustomMaterial(
    texture="Metal",
    tex_name="metal",
    mat_name="MatMetal",
    tex_attrib={"type": "cube"},
    mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
)

tex_attrib = {
    "type": "cube"
}

mat_attrib = {
    "texrepeat": "1 1",
    "specular": "0.4",
    "shininess": "0.1"
}

greenwood = CustomMaterial(
    texture="WoodGreen",
    tex_name="greenwood",
    mat_name="greenwood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)
redwood = CustomMaterial(
    texture="WoodRed",
    tex_name="redwood",
    mat_name="MatRedWood",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)

bluewood = CustomMaterial(
    texture="WoodBlue",
    tex_name="bluewood",
    mat_name="handle1_mat",
    tex_attrib={"type": "cube"},
    mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
)

ceramic = CustomMaterial(
    texture="Ceramic",
    tex_name="ceramic",
    mat_name="MatCeramic",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)
class LiftOtherObjects(ManipulationEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        object=None,
    ):
        self.object = object

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self._assets = {
            'mugbeige': ("3143a4ac", 0.8),          # beige round mug
            'mugbronze': ("34ae0b61", 0.8),          # bronze mug with green inside
            'mugblue': ("128ecbc1", 0.66666667),   # light blue round mug, thicker boundaries
            'mugoffwhite': ("d75af64a", 0.66666667),   # off-white cylindrical tapered mug
            'mugbrown': ("5fe74bab", 0.8),          # brown mug, thin boundaries
            'mugblack': ("345d3e72", 0.66666667),   # black round mug
            'mugred': ("48e260a6", 0.66666667),   # red round mug 
            'mugyellowround': ("8012f52d", 0.8),          # yellow round mug with bigger base 
            'mugyellowcylinder': ("b4ae56d6", 0.8),          # yellow cylindrical mug 
            'mugwooden': ("c2eacc52", 0.8),          # wooden cylindrical mug
            'mugdarkblue': ("e94e46bc", 0.8),          # dark blue cylindrical mug
            'muggreen': ("fad118b3", 0.66666667),   # tall green cylindrical mug
        }

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cube.root_body, target_type="body", return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        if '+' in self.object:
            ingredient_size = [0.03, 0.018, 0.025]
            self.cube = HammerObject(name="hammer",
                                            handle_length=(0.045, 0.05),
                                            handle_radius=(0.012, 0.012),
                                            head_density_ratio=1.0
            )
            self.cabinet_object = CabinetObject(
                name="CabinetObject")
            cabinet_object = self.cabinet_object.get_obj(); cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03))); mujoco_arena.table_body.append(cabinet_object)
            
            for obj_body in [
                    self.cabinet_object,
            ]:
                for material in [lightwood, darkwood, metal, redwood, ceramic]:
                    tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                                    naming_prefix=obj_body.naming_prefix,
                                                                    custom_material=deepcopy(material))
                    obj_body.asset.append(tex_element)
                    obj_body.asset.append(mat_element)

            ingredient_size = [0.015, 0.025, 0.02]
                
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

            self.placement_initializer.append_sampler(
            sampler = UniformRandomSampler(
                name="ObjectSampler-pot",
                mujoco_objects=self.cube,
                x_range=[0.10,  0.18],
                y_range=[-0.20, -0.13],
                rotation=(-0.1, 0.1),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.02,
            ))
            
            mujoco_objects = [
                self.cube,
            ]

            # task includes arena, robot, and objects of interest
            self.model = ManipulationTask(
                mujoco_arena=mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots], 
                mujoco_objects=mujoco_objects,
            )
            self.objects = [
                self.cube,
                self.cabinet_object,
            ]
            self.model.merge_assets(self.cube)
            self.model.merge_assets(self.cabinet_object)

        else:
            self.cube = self.get_class_object(self.object)

            self.cube = BoxObject(
                name="cube",
                size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
                size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
                rgba=[1, 0, 0, 1],
                material=redwood,
            )

            # Create placement initializer
            if self.placement_initializer is not None:
                self.placement_initializer.reset()
                self.placement_initializer.add_objects(self.cube)
            else:
                self.placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    mujoco_objects=self.cube,
                    x_range=[-0.03, 0.03],
                    y_range=[-0.03, 0.03],
                    rotation=None,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                )

            # task includes arena, robot, and objects of interest
            self.model = ManipulationTask(
                mujoco_arena=mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=self.cube,
            )
    def _get_drawer_model(self):
        """
        Allow subclasses to override which drawer to use - should load into @self.drawer.
        """

        # Create drawer object
        tex_attrib = {
            "type": "cube"
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )
        self.drawer = DrawerObject(name="DrawerObject")
        obj_body = self.drawer
        for material in [redwood, ceramic, lightwood]:
            tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                             naming_prefix=obj_body.naming_prefix,
                                                             custom_material=deepcopy(material))
            obj_body.asset.append(tex_element)
            obj_body.asset.append(mat_element)
        return self.drawer
    def get_class_object(self, object):
        if object=='hammer':
            return HammerObject(
                name="hammer",
                handle_length=(0.045, 0.05),
                handle_radius=(0.012, 0.012),
                head_density_ratio=1.0,
            )
        if object=='scatteredhammer':
            return HammerObject(
                name="hammer",
                handle_length=(0.045, 0.05),
                handle_radius=(0.012, 0.012),
                head_density_ratio=1.0,
            )
        elif object=='hammer+drawer':
            return HammerObject(name="hammer",
                    handle_length=(0.045, 0.05),
                    handle_radius=(0.012, 0.012),
                    head_density_ratio=1.0
                ), self._get_drawer_model()
        elif object=='needle':
            return NeedleObject(name="needle_obj")
        elif object=='tripod':
            return RingTripodObject(name="tripod_obj")
        elif object == '3piece':
            self.piece_1_pattern, self.piece_2_pattern, self.base_pattern = self._get_piece_patterns()
            self.piece_1_size = 0.017
            self.piece_2_size = 0.02
            self.base_size = 0.019
            tex_attrib = {
                "type": "cube",
            }
            mat_attrib = {
                "texrepeat": "1 1",
                "specular": "0.4",
                "shininess": "0.1",
            }
            mat = CustomMaterial(
                texture="WoodRed",
                tex_name="redwood",
                mat_name="redwood_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            piece_densities = dict(
                # base=100.,
                # NOTE: changed to make base piece heavier and task easier
                base=10000.,
                piece_1=100.,
                piece_2=100.,
            )
            self.piece_1 = BoxPatternObject(
                name="piece_1",
                unit_size=[self.piece_1_size, self.piece_1_size, self.piece_1_size],
                pattern=self.piece_1_pattern,
                rgba=None,
                material=mat,
                density=piece_densities["piece_1"],
                friction=None,
            )
            self.piece_2 = BoxPatternObject(
                name="piece_2",
                unit_size=[self.piece_2_size, self.piece_2_size, self.piece_2_size],
                pattern=self.piece_2_pattern,
                rgba=None,
                material=mat,
                density=piece_densities["piece_2"],
                friction=None,
                )
            self.base = BoxPatternObject(
                name="base",
                unit_size=[self.base_size, self.base_size, self.base_size],
                pattern=self.base_pattern,
                rgba=None,
                material=mat,
                density=piece_densities["base"],
                friction=None,
            )
            objects = [self.base, self.piece_1, self.piece_2]
            return objects
        elif object=='roundnut':
            return RoundNutObject(name='RoundNut')
        elif object=='squarenut':
            return SquareNutObject(name='SquareNut')
        elif 'mug' in object:
            _shapenet_id, _shapenet_scale = self._assets[object]
            mimicgen_envs_file = '/proj/vondrick3/sruthi/robots/mimicgen_environments/mimicgen_envs/'
            base_mjcf_path = os.path.join(mimicgen_envs_file, "models/robosuite/assets/shapenet_core/mugs")
            mjcf_path = os.path.join(base_mjcf_path, "{}/model.xml".format(_shapenet_id))
            return BlenderObject(
                name="cleanup_object",
                mjcf_path=mjcf_path,
                scale=_shapenet_scale,
                solimp=(0.998, 0.998, 0.001),
                solref=(0.001, 1),
                density=100,
                # friction=(0.95, 0.3, 0.1),
                friction=(1, 1, 1),
                margin=0.001,
            )
        elif 'crp' in object:
            return CRPObject(
                name="crpobj",
            )            
        elif 'cube' in object:
            tex_attrib = {
                "type": "cube",
            }
            mat_attrib = {
                "texrepeat": "1 1",
                "specular": "0.4",
                "shininess": "0.1",
            }
            if object=='greencube':
                greenwood = CustomMaterial(
                    texture="WoodGreen",
                    tex_name="greenwood",
                    mat_name="greenwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                return BoxObject(
                    name="cube",
                    size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
                    size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
                    rgba=[0, 1, 0, 1],
                    material=greenwood,
                )
            elif object=='redcube':
                redwood = CustomMaterial(
                    texture="WoodRed",
                    tex_name="redwood",
                    mat_name="redwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                return BoxObject(
                    name="cube",
                    size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
                    size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
                    rgba=[1, 0, 0, 1],
                    material=redwood,
                )
            elif object=='4wredcube':
                redwood = CustomMaterial(
                    texture="WoodRed",
                    tex_name="redwood",
                    mat_name="redwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                return BoxObject(
                    name="cube",
                    size_min=[0.020, 0.080, 0.020],  # [0.015, 0.015, 0.015],
                    size_max=[0.022, 0.088, 0.022],  # [0.018, 0.018, 0.018])
                    rgba=[1, 0, 0, 1],
                    material=redwood,
                )
            elif object=='25predcube':
                redwood = CustomMaterial(
                    texture="WoodRed",
                    tex_name="redwood",
                    mat_name="redwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                return BoxObject(
                    name="cube",
                    size=[0.021, 0.021, 0.00625],
                    rgba=[1, 0, 0, 1],
                    material=redwood,
                )
            elif object=='10predcube':
                redwood = CustomMaterial(
                    texture="WoodRed",
                    tex_name="redwood",
                    mat_name="redwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                return BoxObject(
                    name="cube",
                    size=[0.021, 0.021, 0.003125],
                    rgba=[1, 0, 0, 1],
                    material=redwood,
                )
            elif object=='50predcube':
                redwood = CustomMaterial(
                    texture="WoodRed",
                    tex_name="redwood",
                    mat_name="redwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                return BoxObject(
                    name="cube",
                    size=[0.021, 0.021, 0.0125],
                    rgba=[1, 0, 0, 1],
                    material=redwood,
                )

        assert True==False
        

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            sensors = [cube_pos, cube_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to cube position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04
