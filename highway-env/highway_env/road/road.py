# cython: profile=True

from commonroad.common.util import enum
import numpy as np
import logging
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet

from highway_env.road.lane import (
    LineType,
    StraightLane,
    AbstractLane,
    CommonRoadLane,
)
from highway_env.road.objects import Landmark
from highway_env import utils
from shapely.lib import is_ring

if TYPE_CHECKING:
    from highway_env.vehicle import kinematics
    from highway_env.road import objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class RoadNetworkCommonRoad(object):
    lanelet_network: LaneletNetwork

    def __init__(self, net, is_ring=False):
        self.lanelet_network = net

        self.lanes = dict()
        ids = self.lanelet_network._lanelets.keys()

        self.lane_ids = np.array(list(ids), dtype=int)

        # create our custom lane definitions
        for id in ids:
            self.lanes[id] = CommonRoadLane(
                self.lanelet_network.find_lanelet_by_id(id), is_ring
            )

    def get_lane(self, index: int) -> CommonRoadLane:
        """
        Get the lanelet_network corresponding to a given index in the road network.

        :param index: id of the lanelet.
        :return: the corresponding lanelet.
        """
        return self.lanes[index]

    def get_closest_lane_index(self, point: np.ndarray, heading=None) -> int:
        """
        Get the the lane closest to a world position.

        :param point: a world position [m].
        :return: the closest lane.
        """

        # distances for sorting
        distance_list = np.zeros(len(self.lane_ids), dtype=np.float32)

        # go through list of lanelets
        for i, id in enumerate(self.lane_ids):
            lanelet = self.get_lane(id)

            # compute minimum distances to each road
            distance_list[i] = utils.pymindist(
                lanelet.lengths, lanelet.lanelet.center_vertices, point
            )

        # get lanelet with smallest distance
        min_index = distance_list.argmin()
        return self.lane_ids[min_index]

    def next_lane(
        self,
        current_index: int,
        route: Route = None,
        position: np.ndarray = None,
        np_random: np.random.RandomState = np.random,
    ) -> Optional[int]:
        """
        Get the index of the next lane that should be followed after finishing the current lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        # Assumption only one successor
        successors = self.lanelet_network.find_lanelet_by_id(current_index).successor
        if len(successors) > 0:
            return self.lanelet_network.find_lanelet_by_id(current_index).successor[0]
        # if there is no successor road just follow the current on
        return self.lanelet_network.find_lanelet_by_id(current_index).lanelet_id

    def side_lanes(self, lane_index: int) -> List[Lanelet]:
        """
        :param lane_index: the index of a lane.
        :return: indexes of lanes next to a an input lane, to its right or left.
        """
        lanelet = self.lanelet_network.find_lanelet_by_id(lane_index)
        ids = []
        if lanelet.adj_left is not None:
            ids.append(lanelet.adj_left)
        if lanelet.adj_right is not None:
            ids.append(lanelet.adj_right)
        return ids


class RoadNetwork(object):
    graph: Dict[str, Dict[str, List[AbstractLane]]]
    lane_indices: List[LaneIndex]
    lanes: List[AbstractLane]

    lane_indices = []

    def __init__(self):
        self.graph = {}
        self.lane_indices = []
        self.lanes = []

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(
        self, position: np.ndarray, heading: Optional[float] = None
    ) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """

        if not self.lane_indices:
            for _from, to_dict in self.graph.items():
                for _to, lanes in to_dict.items():
                    for _id, l in enumerate(lanes):
                        self.lane_indices.append((_from, _to, _id))
                        self.lanes.append(self.get_lane((_from, _to, _id)))

        current_smallest = 1e8
        l_len = len(self.lane_indices)

        for i in range(l_len):
            index = self.lane_indices[i]
            lane = self.lanes[i]
            curr_dist = lane.distance_with_heading(position, heading)
            if curr_dist < current_smallest:
                current_smallest = curr_dist
                closest_lane_index = index

        # print("Done \n\n")

        return closest_lane_index

    def next_lane(
        self,
        current_index: LaneIndex,
        route: Route = None,
        position: np.ndarray = None,
        np_random: np.random.RandomState = np.random,
    ) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick closest next road
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = None
        # Pick next road according to planned route
        if route:
            if (
                route[0][:2] == current_index[:2]
            ):  # We just finished the first step of the route, drop it.
                route.pop(0)
            if (
                route and route[0][0] == _to
            ):  # Next road in route is starting at the end of current road.
                _, next_to, _ = route[0]
            # elif route:
            # logger.warning("Route {} does not start after current road {}.".format(route[0], current_index))

        # pick closest next road
        if not next_to:
            try:
                # next_to = list(self.graph[_to].keys())[np.random.randint(len(self.graph[_to]))]

                lane_candidates = []
                for next_to in list(self.graph[_to].keys()):
                    for l in range(len(self.graph[_to][next_to])):
                        lane_candidates.append((_to, next_to, l))

                next_lane = min(
                    lane_candidates, key=lambda l: self.get_lane(l).distance(position)
                )
                return next_lane
            except KeyError:
                return current_index

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: indexes of lanes next to a an input lane, to its right or left.
        """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 in the same road as lane 2?"""
        return lane_index_1[:2] == lane_index_2[:2] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    @staticmethod
    def is_leading_to_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 leading to of lane 2?"""
        return lane_index_1[1] == lane_index_2[0] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    def is_connected_road(
        self,
        lane_index_1: LaneIndex,
        lane_index_2: LaneIndex,
        route: Route = None,
        same_lane: bool = False,
        depth: int = 0,
    ) -> bool:
        """
        Is the lane 2 leading to a road within lane 1's route?

        Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if RoadNetwork.is_same_road(
            lane_index_2, lane_index_1, same_lane
        ) or RoadNetwork.is_leading_to_road(lane_index_1, lane_index_2, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(
                    lane_index_1, lane_index_2, route[1:], same_lane, depth
                )
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(
                    route[0], lane_index_2, route[1:], same_lane, depth - 1
                )
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                # return any([self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
                #             for l1_to in self.graph.get(_to, {}).keys()])
                l1_to = list(self.graph.get(_to, {}).keys())[0]
                return self.is_connected_road(
                    (_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1
                )
        return False

    def get_connected_route(
        self,
        lane_index_1: LaneIndex,
        lane_index_2: LaneIndex,
        route: Route,
        same_lane: bool = False,
        depth: int = 0,
    ):
        ret_route = []
        lane1 = lane_index_1
        ret_route.append(lane1)

        while not (
            RoadNetwork.is_same_road(lane_index_2, lane1, same_lane)
            or RoadNetwork.is_leading_to_road(lane1, lane_index_2, same_lane)
        ):
            _from, _to, _id = lane1
            l1_to = list(self.graph.get(_to, {}).keys())[0]
            lane1 = (_to, l1_to, _id)
            ret_route.append(lane1)

        ret_route.append(lane_index_2)
        return ret_route


class RoadCommonRoad(object):
    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(
        self,
        network: RoadNetworkCommonRoad = None,
        vehicles: List["kinematics.Vehicle"] = None,
        road_objects: List["objects.RoadObject"] = None,
        np_random: np.random.RandomState = None,
        record_history: bool = False,
    ) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_vehicles_to(
        self,
        vehicle: "kinematics.Vehicle",
        distance: float,
        count: int = None,
        see_behind: bool = True,
    ) -> object:
        distance = (
            distance**2
        )  # need to square it, because hacky norm does not use sqrt
        vehicles = [
            v
            for v in self.vehicles
            if utils.norm(v.position, vehicle.position) < distance
            and v is not vehicle
            and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))
        ]

        # vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        vehicles = sorted(vehicles, key=lambda v: abs(v.lane_distance_to(vehicle)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        # e.g., len(self.vehicles) = 7
        # if vehicle: IDMVehicle, it will go to the behavior.py
        # if vehicle: MDPVehicle, it will go to the behavior.py
        for vehicle in self.vehicles:  # all the vehicles on the road
            vehicle.act()

    def step(self, dt) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        vehicles = self.vehicles
        objects = self.objects
        len_v = len(vehicles)
        len_o = len(objects)

        for i in range(len_v):
            vehicles[i].step(dt)

        # TODO check collision only every Xth step
        # TODO collect all vehicle positions and check collision at once
        for i in range(len_v):
            v = vehicles[i]
            for j in range(len_v):
                v.check_collision(vehicles[j])
            for j in range(len_o):
                v.check_collision(objects[j])

        # remove vehicles that reached the end of the road
        for v in self.vehicles:
            if (
                self.network.get_lane(v.lane_index).after_end(
                    v.position, vehicle_length=v.LENGTH
                )
                and v.lane_index == 5
            ):
                # print(f"{v.id} reached the end")
                self.vehicles.remove(v)

    def surrounding_vehicles(
        self, vehicle: "kinematics.Vehicle", lane_index: Optional[int] = None
    ) -> Tuple[Optional["kinematics.Vehicle"], Optional["kinematics.Vehicle"]]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)

        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if v is not vehicle:  # and not isinstance(v, Landmark):
                if not (
                    v.lane_index == lane_index
                    # or lane_index
                    # == lane.lanelet.adj_right  # check left and right of the lane we want to change to also to avoid to vehicles changing to the same lane and crashing
                    # or lane_index == lane.lanelet.adj_left
                ):
                    continue

                d = lane.distance_between_points(vehicle.position, v.position)

                if d >= 0 and (s_front is None or abs(d) <= s_front):
                    s_front = d
                    v_front = v
                if d < 0 and (s_rear is None or abs(d) < abs(s_rear)):
                    s_rear = d
                    v_rear = v

        return v_front, v_rear


class Road(object):
    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(
        self,
        network: RoadNetwork = None,
        vehicles: List["kinematics.Vehicle"] = None,
        road_objects: List["objects.RoadObject"] = None,
        np_random: np.random.RandomState = None,
        record_history: bool = False,
    ) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_vehicles_to(
        self,
        vehicle: "kinematics.Vehicle",
        distance: float,
        count: int = None,
        see_behind: bool = True,
    ) -> object:
        distance = (
            distance**2
        )  # need to square it, because hacky norm does not use sqrt
        vehicles = [
            v
            for v in self.vehicles
            # if np.linalg.norm(v.position - vehicle.position) < distance
            if utils.norm(v.position, vehicle.position) < distance
            and v is not vehicle
            and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))
        ]

        vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        # e.g., len(self.vehicles) = 7
        # if vehicle: IDMVehicle, it will go to the behavior.py
        # if vehicle: MDPVehicle, it will go to the behavior.py
        for vehicle in self.vehicles:  # all the vehicles on the road
            vehicle.act()

    def step(self, dt) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        vehicles = self.vehicles
        objects = self.objects
        # cdef int i, j
        # cdef int len_v = len(vehicles)
        # cdef int len_o = len(objects)
        len_v = len(vehicles)
        len_o = len(objects)

        for i in range(len_v):
            vehicles[i].step(dt)
        # for vehicle in self.vehicles:
        # vehicle.step(dt)

        for i in range(len_v):
            v = vehicles[i]
            for j in range(len_v):
                v.check_collision(vehicles[j])
            for j in range(len_o):
                v.check_collision(objects[j])
        # for vehicle in self.vehicles:
        # for other in self.vehicles:
        # vehicle.check_collision(other)
        # for other in self.objects:
        # vehicle.check_collision(other)

    def surrounding_vehicles(
        self, vehicle: "kinematics.Vehicle", lane_index: LaneIndex = None
    ) -> Tuple[Optional["kinematics.Vehicle"], Optional["kinematics.Vehicle"]]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if (
                v is not vehicle and not isinstance(v, Landmark)
            ):  # self.network.is_connected_road(v.lane_index, lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)

                d = lane.distance_between_points(vehicle.position, v.position)

                # if vehicle.id == 4 and v.id == 2:
                #     print(f"2 has lane {v.lane_index} 4 has lane {lane_index}")
                #     print(f"on_lane {lane.on_lane(v.position, float(s_v), float(lat_v), margin=0.05)}")

                # if not lane.on_lane(v.position, float(s_v), float(lat_v), margin=0.05) and not self.network.is_connected_road(v.lane_index, lane_index, same_lane=True) \
                # if not self.network.is_connected_road(
                #     v.lane_index, lane_index, same_lane=True
                # ) and not self.network.is_connected_road(
                #     lane_index, v.lane_index, same_lane=True
                # ):
                #     # if vehicle.id == 3:
                #     #     print(f"skipping {v} with {vehicle.lane_index} and {v.lane_index}")
                #     continue
                if d >= 0 and (s_front is None or d <= s_front):
                    s_front = d
                    v_front = v
                if d < 0 and (s_rear is None or d > s_rear):
                    s_rear = d
                    v_rear = v

        return v_front, v_rear

        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        # cdef float s_v
        # cdef float s = vehicle.position[0]  # x position
        s_front = s_rear = None
        v_front = v_rear = None

        from highway_env.vehicle.kinematics import RealVehicle

        # we do not consider obstacles
        for v in self.vehicles:
            # if v is not vehicle:# and not isinstance(v, Landmark):
            if v is not vehicle and not isinstance(v, RealVehicle):
                if (
                    lane_index == ("a", "b", 0)
                    or lane_index == ("b", "c", 0)
                    or lane_index == ("c", "d", 0)
                ):
                    if lane_index == ("a", "b", 0) and (
                        v.lane_index == ("a", "b", 0) or v.lane_index == ("b", "c", 0)
                    ):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    elif lane_index == ("b", "c", 0) and (
                        v.lane_index == ("a", "b", 0)
                        or v.lane_index == ("b", "c", 0)
                        or v.lane_index == ("c", "d", 0)
                    ):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    elif lane_index == ("c", "d", 0) and (
                        v.lane_index == ("b", "c", 0) or v.lane_index == ("c", "d", 0)
                    ):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    else:
                        continue
                elif (
                    lane_index == ("a", "b", 1)
                    or lane_index == ("b", "c", 1)
                    or lane_index == ("c", "d", 1)
                ):
                    if lane_index == ("a", "b", 1) and (
                        v.lane_index == ("a", "b", 1) or v.lane_index == ("b", "c", 1)
                    ):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    elif lane_index == ("b", "c", 1) and (
                        v.lane_index == ("a", "b", 1)
                        or v.lane_index == ("b", "c", 1)
                        or v.lane_index == ("c", "d", 1)
                    ):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    elif lane_index == ("c", "d", 1) and (
                        v.lane_index == ("b", "c", 1) or v.lane_index == ("c", "d", 1)
                    ):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    else:
                        continue
                elif lane_index == ("b", "c", 2):
                    if v.lane_index == ("b", "c", 2) or v.lane_index == ("k", "b", 0):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    else:
                        continue
                elif lane_index == ("c", "o", 0):
                    if v.lane_index == ("c", "o", 0) or v.lane_index == ("b", "c", 2):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    else:
                        continue
                else:
                    if lane_index == ("j", "k", 0) and (
                        v.lane_index == ("j", "k", 0) or v.lane_index == ("k", "b", 0)
                    ):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    elif lane_index == ("k", "b", 0) and (
                        v.lane_index == ("j", "k", 0)
                        or v.lane_index == ("k", "b", 0)
                        or v.lane_index == ("b", "c", 1)
                    ):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    elif lane_index == ("b", "c", 1) and (
                        v.lane_index == ("k", "b", 0) or v.lane_index == ("b", "c", 1)
                    ):
                        # s_v, lat_v = v.position
                        s_v = v.position[0]
                    else:
                        continue

                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    # @profile
    def neighbour_vehicles(
        self, vehicle: "kinematics.Vehicle", lane_index: LaneIndex = None
    ) -> Tuple[Optional["kinematics.Vehicle"], Optional["kinematics.Vehicle"]]:
        """
        Find the preceding and following vehicles of a given vehicle.
        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(
                v, Landmark
            ):  # self.network.is_connected_road(v.lane_index,
                # lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def __repr__(self):
        return self.vehicles.__repr__()
