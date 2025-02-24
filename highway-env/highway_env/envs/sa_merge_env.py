import numpy as np

# from gym.envs.registration import register
from gymnasium.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import (
    LineType,
    StraightLane,
    SineLane,
    HorizontalLane,
    DEFAULT_WIDTH,
)
from highway_env.road.road import (
    Road,
    RoadNetwork,
    RoadNetworkCommonRoad,
    RoadCommonRoad,
)
from highway_env.vehicle.behavior import ModelIDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.graphics import VehicleGraphics

# from highway_env.vehicle.objects import Obstacle
from highway_env.road.objects import Obstacle

from highway_env.vehicle.kinematics import Vehicle, RealVehicle

from commonroad.common.file_reader import CommonRoadFileReader


class SingleAgentMergeEnv(AbstractEnv):
    """
    A highway-env merge negotiation environment.

    The ego-vehicle is driving on a highway-env and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    n_a = 5
    n_s = (25,)

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "duration": 15,  # time step
                "policy_frequency": 5,  # [Hz]
                "merging_speed_reward": -0.5,
                "right_lane_reward": 0.1,
                "lane_change_reward": -0.05,
                "reward_speed_range": [10, 30],
                "collision_reward": 200,
                "high_speed_reward": 1,
                "offramp_reward": 100,
                "HEADWAY_COST": 4,  # default=1
                # "HEADWAY_COST": 1,  # default=1
                "HEADWAY_TIME": 1.2,  # default=1.2[s]
                "MERGING_LANE_COST": 4,  # default=4
                "LANE_CHANGE_COST": 1,  # default=0.5
                # "LANE_CHANGE_COST": 0.5,  # default=0.5
                "traffic_density": 1,  # easy or hard modes
                "scaling": 320.76,
                # "scaling": 389.402,
                # "scaling": 136.34,
                # "centering_position": [0.8, -0.8],
                "centering_position": [0.8, -0.7],
            }
        )
        return cfg

    def set_vehicle(self, veh):
        self.vehicle = veh

    def _reward(self, action: int) -> float:
        # Cooperative reward
        return self._agent_reward(action, self.vehicle)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        # the optimal reward is 0
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        # compute cost for staying on the merging lane
        if (
            vehicle.lane_index == 7146164179188
            and vehicle.position[0] > 230
            and vehicle.position[0] < 320
        ):
            Merging_lane_cost = -np.exp(
                -((vehicle.position[0] - sum(self.ends[:3])) ** 2) / (10 * self.ends[2])
            )
        else:
            Merging_lane_cost = 0

        # give penalty if the agent drives on the offramp
        if vehicle.lane_index == 7146164179188 and vehicle.position[0] > 320:
            offramp_cost = -self.config["offramp_reward"]
        else:
            offramp_cost = 0

        # lane change cost to avoid unnecessary/frequent lane changes
        Lane_change_cost = (
            -1 * self.config["LANE_CHANGE_COST"] if action == 0 or action == 2 else 0
        )
        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = (
            np.log(headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed))
            if vehicle.speed > 0
            else 0
        )

        # compute overall reward
        reward = (
            self.config["collision_reward"] * (-1 * vehicle.crashed)
            + (self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1))
            + self.config["MERGING_LANE_COST"] * Merging_lane_cost
            + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
            + Lane_change_cost
            + offramp_cost
        )

        return reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        # return self.vehicle.crashed or self.vehicle.position[0] > 370 #or self.steps >= 200
        # return (
        #     self.vehicle.crashed
        #     or self.vehicle.position[0] > 400
        #     or self.vehicle.lane_index == 7146164179188
        #     and self.vehicle.position[0] > 320
        # )  # end the episode if the vehicle drives off ramp
        return (
            self.vehicle.crashed
            # or self.vehicle.position[0] > 14.2
            or self.vehicle.position[0] > 3.0
            # or self.vehicle.position[1] > 1.4
            or self.vehicle.position[1] > 0.5
            # or self.vehicle.lane_index == 31371651486
            # or self.vehicle.lane_index == 1568
            # or self.vehicle.lane_index == 1567
            # and self.vehicle.position[0] > 11.42
        )  # end the episode if the vehicle drives off ramp
        # return self.vehicle.crashed \
        # or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _reset(self, num_CAV=1, num_HDV=6) -> None:
        self._make_road()
        if num_CAV != 1:
            num_CAV = 1
        else:
            num_CAV = num_CAV
        if self.config["traffic_density"] == 1:
            # easy mode: 6-8 HDVs
            num_HDV = np.random.choice(np.arange(6, 9), 1)[0]
        elif self.config["traffic_density"] == 2:
            # easy mode: 9-12 DVs
            num_HDV = np.random.choice(np.arange(9, 13), 1)[0]
        elif self.config["traffic_density"] == 3:
            # easy mode: 13-15 HDVs
            num_HDV = np.random.choice(np.arange(13, 16), 1)[0]
            # num_HDV = np.random.choice(np.arange(16, 19), 1)[0]
        self._make_vehicles(num_CAV, num_HDV)
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway-env and a merging lane.

        :return: the road
        """

        scenario, _ = CommonRoadFileReader(
            # "./DEU_HighwayMergeConnected2longer-1_1_T-1.xml"
            # "./DEU_MergeCircCont-1_1_T-1.xml"
            #
            # "./DEU_MergeCircContLonger-1_1_T-1.xml"
            #
            # "./DEU_ScaledHighway-1_1_T-1.xml"
            # "./DEU_tightercircularmerge-1_1_T-1.xml"
            "./DEU_zerocentercircmerge-1_1_T-1.xml"
        ).open()

        net = scenario.lanelet_network
        # net_common_road = RoadNetworkCommonRoad(net, is_ring=False)
        net_common_road = RoadNetworkCommonRoad(net, is_ring=True)

        self.road = RoadCommonRoad(network=net_common_road)

        # net = RoadNetwork()
        #
        # # Highway lanes
        # # self.ends = [150, 80, 200, 150]  # Before, converging, merge, after
        self.ends = [150, 80, 80, 150]  # Before, converging, merge, after
        # # self.ends = [150, 80, 40, 40, 150]  # Before, converging, merge, after
        #
        for i in range(len(self.ends)):
            self.ends[i] /= 28

        #
        # c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        # y = [0, DEFAULT_WIDTH]
        # line_type = [(c, s), (n, c)]
        # line_type_merge = [(c, s), (n, s)]
        # for i in range(2):
        #     net.add_lane("a", "b", HorizontalLane([0, y[i]], [sum(self.ends[:2]), y[i]], line_types=line_type[i]))
        #     net.add_lane("b", "c", HorizontalLane([sum(self.ends[:2]), y[i]], [sum(self.ends[:3]), y[i]], line_types=line_type_merge[i]))
        #     net.add_lane("c", "d", HorizontalLane([sum(self.ends[:3]), y[i]], [sum(self.ends[:4]), y[i]], line_types=line_type_merge[i]))
        #     # net.add_lane("d", "e", HorizontalLane([sum(self.ends[:4]), y[i]], [sum(self.ends), y[i]], line_types=line_type[i]))
        #
        # # Merging lane
        # amplitude = 3.25
        # ljk = HorizontalLane([0, 6.5 + 4 + 4], [self.ends[0], 6.5 + 4 + 4], line_types=(c, c), forbidden=True)
        #
        # lkb = SineLane(ljk.position(self.ends[0], -amplitude), ljk.position(sum(self.ends[:2]), -amplitude),
        #                amplitude, 2 * np.pi / (2*self.ends[1]), np.pi / 2, line_types=(c, c), forbidden=True)
        #
        # lbc = HorizontalLane(lkb.position(self.ends[1], 0), lkb.position(self.ends[1], 0) + [self.ends[2], 0],
        #                    line_types=(n, c), forbidden=False)
        # # lcd = HorizontalLane(lbc.position(self.ends[2], 0), lbc.position(self.ends[2], 0) + [self.ends[3], 0],
        #                    # line_types=[n, c], forbidden=True)
        # #off ramp
        # # lco = StraightLane(lcd.position(self.ends[2], 0), lcd.position(self.ends[2]+80, 6.5), line_types=[c, c], forbidden=True)
        # lco = StraightLane(lbc.position(self.ends[2], 0), lbc.position(self.ends[2]+80, 6.5), line_types=(c, c), forbidden=True)
        # lou = HorizontalLane([sum(self.ends[:3])+80, 6.5+4+4], [sum(self.ends[:3])+80+70, 6.5+4+4], line_types=(c, c), forbidden=True)
        #
        # net.add_lane("j", "k", ljk)
        # net.add_lane("k", "b", lkb)
        # net.add_lane("b", "c", lbc)
        # net.add_lane("c", "o", lco)
        # # net.add_lane("d", "o", ldo) #off ramp
        # net.add_lane("o", "u", lou) #off ramp
        # road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        # # road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))
        # self.road = road

    def _make_vehicles(self, num_CAV=1, num_HDV=3) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """

        num_HDV = 15
        road = self.road
        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_vehicles_type = ModelIDMVehicle

        spawn_points_s1 = [10, 50, 90, 130, 170, 210, 225]
        spawn_points_s2 = [0, 40, 80, 120, 160, 200, 220]
        # spawn_points_m = [5, 45, 85, 125, 130, 145, 150, 165, 205, 225]
        # spawn_points_m = [5, 45, 85, 125, 165, 205, 225]
        spawn_points_m = [0, 0.6, 1, 1.5, 2, 2.5]  # adjusted for circular track
        # spawn_points_m_cav = [125, 130, 145, 150, 165]
        # spawn_points_m_cav = [125, 165]
        spawn_points_m_cav = [0, 0.6]  # adjusted for merge

        for i in range(len(spawn_points_s1)):
            spawn_points_s1[i] /= 28
        for i in range(len(spawn_points_s2)):
            spawn_points_s2[i] /= 28
        # for i in range(len(spawn_points_m)):
        #     spawn_points_m[i] /= 28
        # for i in range(len(spawn_points_m_cav)):
        #     spawn_points_m_cav[i] /= 28

        # merging_lane_index = 7146164179188
        # through_lane1_index = 1
        # through_lane2_index = 3
        # merging_lane_index = 12371236
        #
        #
        # through_lane1_index = 1235
        # through_lane2_index = 1234
        # merging_lane_index = 12371236

        # merging_lane_index = 1569
        # through_lane1_index = 1567
        # through_lane2_index = 1568

        merging_lane_index = 5
        through_lane1_index = 1234
        through_lane2_index = 1235

        # initial speed with noise and location noise
        initial_speed = (
            np.random.rand(num_CAV + num_HDV) * 8 + 22
        )  # range from [25, 30]
        # initial_speed /= 28  # scale to model vehicles
        initial_speed /= 50  # scale to real model vehicles
        initial_speed = list(initial_speed)

        # loc_noise = np.random.rand(num_CAV + num_HDV) * 6 - 3  # range from [-1.5, 1.5]
        loc_noise = np.random.rand(num_CAV + num_HDV) / 10  # range from [-1.5, 1.5]
        loc_noise = list(loc_noise)

        """Spawn points for CAV"""
        spawn_point_m_c = np.random.choice(spawn_points_m_cav, num_CAV, replace=False)
        spawn_point_m_c = list(spawn_point_m_c)

        for c in spawn_point_m_c:
            spawn_points_m.remove(c)
        """spawn the rest CAV on the merging road"""
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(merging_lane_index).position(
                spawn_point_m_c.pop(0) + loc_noise.pop(0), 0
            ),
            speed=initial_speed.pop(0),
        )
        self.vehicle = ego_vehicle
        road.vehicles.append(ego_vehicle)

        self.vehicle.color = (200, 0, 150)
        self.vehicle.id = 0
        # return

        """Spawn points for HDV"""
        # spawn point indexes on the straight road
        spawn_point_s_h1 = np.random.choice(
            spawn_points_s1, num_HDV // 3, replace=False
        )
        spawn_point_s_h2 = np.random.choice(
            spawn_points_s2, num_HDV // 3, replace=False
        )
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(
            spawn_points_m, num_HDV - 2 * num_HDV // 3, replace=False
        )
        spawn_point_s_h1 = list(spawn_point_s_h1)
        spawn_point_s_h2 = list(spawn_point_s_h2)
        spawn_point_m_h = list(spawn_point_m_h)

        right_bias = 8.0
        offramp_percentage = 0.3
        biases = list(
            np.random.choice(
                [-right_bias, right_bias],
                num_HDV,
                p=[1 - offramp_percentage, offramp_percentage],
            )
        )

        """spawn the HDV on the main road first"""
        for _ in range(num_HDV // 3):
            veh = other_vehicles_type(
                road,
                road.network.get_lane(through_lane1_index).position(
                    spawn_point_s_h1.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
            )

            veh.RIGHT_BIAS = biases.pop(0)
            veh.color = (
                VehicleGraphics.BLUE
                if veh.RIGHT_BIAS == right_bias
                else VehicleGraphics.GREEN
            )
            road.vehicles.append(veh)

        for _ in range(num_HDV // 3):
            veh = other_vehicles_type(
                road,
                road.network.get_lane(through_lane2_index).position(
                    spawn_point_s_h2.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
            )

            veh.RIGHT_BIAS = biases.pop(0)
            veh.color = (
                VehicleGraphics.BLUE
                if veh.RIGHT_BIAS == right_bias
                else VehicleGraphics.GREEN
            )
            road.vehicles.append(veh)

        """spawn the rest HDV on the merging road"""
        for _ in range(num_HDV - 2 * num_HDV // 3):
            # for _ in range(num_HDV // 5):
            veh = other_vehicles_type(
                road,
                road.network.get_lane(merging_lane_index).position(
                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
            )

            # all merging vehicles want on main road (left bias)
            veh.RIGHT_BIAS = -4.0
            veh.color = VehicleGraphics.GREEN
            road.vehicles.append(veh)

        """ create the vehicle from real data """
        # traj_list = [69, 79, 80, 81, 85, 87, 91, 92, 100, 101, 103, 104, 105, 114, 115, 117, 119, 122, 125, 129, 131]
        # traj_list = [69, 79, 80, 81, 85, 87, 91, 92, 100, 101, 103, 104, 105, 114, 115, 117, 119, 122, 125, 129, 131, 133, 135, 138, 145, 156, 160]
        # traj_list = [69]

        # for traj in traj_list:
        # v = RealVehicle(f"../traj{traj}.npy", 0)
        # v.color = VehicleGraphics.BLACK
        # road.vehicles.append(v)
        # v.id = traj


register(
    id="merge-single-agent-v0",
    entry_point="highway_env.envs:SingleAgentMergeEnv",
)
