# cython: language_level=3, cdivision = True, profile=True
from typing import Union, TYPE_CHECKING, Optional
import numpy as np
from collections import deque
import math

from highway_env import utils
from highway_env.road.lane import AbstractLane
from highway_env.road.road import Road, LaneIndex
from highway_env.road.objects import Obstacle, Landmark
from highway_env.types import Vector

from scipy import interpolate


if TYPE_CHECKING:
    from highway_env.road.objects import RoadObject


class Vehicle(object):
    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 5.0
    LENGTH_SQUARE = LENGTH**2  # Nedded for faster distance comparison
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.0
    """ Maximum reachable speed [m/s] """

    def __init__(
        self, road: Road, position: Vector, heading: float = 0.0, speed: float = 0.0
    ):
        self.road = road
        # self.position = np.array(position, dtype=float)
        self.position = position
        self.heading = heading
        self.speed = speed
        self.lane_index = (
            self.road.network.get_closest_lane_index(self.position, float(self.heading))
            if self.road
            else np.nan
        )

        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {"steering": 0, "acceleration": 0}
        self.trajectories = []
        self.crashed = False
        self.log = []
        self.local_reward = 0
        self.regional_reward = 0
        self.history = deque(maxlen=30)

    @classmethod
    def make_on_lane(
        cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0
    ) -> "Vehicle":
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param speed: initial speed in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        if speed is None:
            speed = lane.speed_limit
        return cls(
            road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed
        )

    @classmethod
    def create_random(
        cls,
        road: Road,
        speed: float = None,
        lane_id: Optional[int] = None,
        spacing: float = 1,
    ) -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        if speed is None:
            speed = road.np_random.uniform(
                Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1]
            )
        default_spacing = 1.5 * speed
        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = (
            lane_id
            if lane_id is not None
            else road.np_random.choice(len(road.network.graph[_from][_to]))
        )
        lane = road.network.get_lane((_from, _to, _id))
        offset = (
            spacing
            * default_spacing
            * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        )
        x0 = (
            np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            if len(road.vehicles)
            else 3 * offset
        )
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_from(cls, vehicle: "Vehicle") -> "Vehicle":
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        return v

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    # @profile
    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action["steering"]
        beta = math.atan(0.5 * math.tan(delta_f))
        v = self.speed * np.array(
            [math.cos(self.heading + beta), math.sin(self.heading + beta)]
        )
        self.position += v * dt
        self.heading += self.speed * math.sin(beta) / (self.LENGTH / 2) * dt
        self.heading = utils.wrap_to_pi(self.heading)
        self.speed += self.action["acceleration"] * dt
        self.speed = max(self.speed, 0.18)
        self.on_state_update()

    def clip_actions(self) -> None:
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = -1.0 * self.speed
        self.action["steering"] = float(self.action["steering"])
        self.action["acceleration"] = float(self.action["acceleration"])
        if self.speed > self.MAX_SPEED:
            self.action["acceleration"] = min(
                self.action["acceleration"], 1.0 * (self.MAX_SPEED - self.speed)
            )
        elif self.speed < -self.MAX_SPEED:
            self.action["acceleration"] = max(
                self.action["acceleration"], 1.0 * (self.MAX_SPEED - self.speed)
            )

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(
                self.position, float(self.heading)
            )
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def lane_distance_to(self, vehicle: "Vehicle", lane: AbstractLane = None) -> float:
        """
        Compute the signed distance to another vehicle along a lane.

        :param vehicle: the other vehicle
        :param lane: a lane
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane

        return lane.distance_between_points(self.position, vehicle.position)

    def check_collision(self, other: Union["Vehicle", "RoadObject"]) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        """
        if self.crashed or other is self:
            return

        if isinstance(other, Vehicle):
            if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED:
                return

            if self._is_colliding(other):
                self.speed = other.speed = min([self.speed, other.speed], key=abs)
                self.crashed = other.crashed = True
        elif isinstance(other, Obstacle):
            if not self.COLLISIONS_ENABLED:
                return

            if self._is_colliding(other):
                self.speed = min([self.speed, 0], key=abs)
                self.crashed = other.hit = True
        elif isinstance(other, Landmark):
            if self._is_colliding(other):
                other.hit = True

    def _is_colliding(self, other):
        # Fast spherical pre-check
        if utils.norm(other.position, self.position) > self.LENGTH_SQUARE:
            return False
        # Accurate rectangular check
        rect = utils.middle_to_vertices(
            self.position, self.LENGTH, self.WIDTH, self.heading
        )
        other_rect = utils.middle_to_vertices(
            other.position, other.LENGTH, other.WIDTH, other.heading
        )

        return utils.separating_axis_theorem(rect, other_rect)

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            # last_lane = self.road.network.get_lane(self.route[-1])
            last_lane_index = self.route[-1]
            last_lane_index = (
                last_lane_index
                if last_lane_index[-1] is not None
                else (*last_lane_index[:-1], 0)
            )
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(
                self.destination - self.position
            )
        else:
            return np.zeros((2,))

    @property
    def on_road(self) -> bool:
        """Is the vehicle on its current lane, or off-road ?"""
        return self.lane.on_lane(self.position)

    def front_distance_to(self, other: "Vehicle") -> float:
        return self.direction.dot(other.position - self.position)

    def to_dict(
        self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True
    ) -> dict:
        vel = self.velocity
        d = {
            "presence": 1,
            "x": self.position[0],
            "y": self.position[1],
            "vx": vel[0],
            "vy": vel[1],
            "heading": self.heading,
            #'cos_h': self.direction[0],
            #'sin_h': self.direction[1],
            #'cos_d': self.destination_direction[0],
            #'sin_d': self.destination_direction[1]
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            # print(f"{origin_dict=}")
            # print(f"{d=}")
            for key in ["x", "y", "vx", "vy"]:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        # return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)
        return f"#{self.id}"

    def __repr__(self):
        return self.__str__()


class RealVehicle(Vehicle):
    def __init__(self, traj_file: str, start_time: int):
        self.lane = None
        self.lane_index = None

        self.traj = np.load(traj_file)
        self.position = np.zeros(
            2,
        )
        self.crashed = False

        # Order of each row is: time, x_pos, y_pos, speed in kph, heading
        self.position[0] = self.traj[0][1]
        self.position[1] = self.traj[0][2]
        self.speed = self.traj[0][3]
        self.heading = self.traj[0][4]

        self.traj[:, 0] -= start_time

        self.fx = interpolate.interp1d(self.traj[:, 0], self.traj[:, 1])
        self.fy = interpolate.interp1d(self.traj[:, 0], self.traj[:, 2])
        self.fv = interpolate.interp1d(self.traj[:, 0], self.traj[:, 3])
        self.fa = interpolate.interp1d(self.traj[:, 0], self.traj[:, 4])

        self.time = 0

    def step(self, dt: float) -> None:
        self.time += dt

        try:
            self.position[0] = self.fx(self.time)
            self.position[1] = self.fy(self.time)
            self.speed = self.fv(self.time) / 3.6  # Conversion from kph to m/s
            self.heading = self.fa(self.time)
        except ValueError:
            # print("Value outside interpolation limit")
            pass

    def act(self):
        pass
