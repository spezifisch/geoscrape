#!/usr/bin/python3

import logging
import numpy as np
import math

from s2sphere import *

try:
    from pgoapi import utilities as util
    PGO = True
except ImportError:
    PGO = False

log = logging.getLogger(__name__)

def to_radians(theta):
    return np.divide(np.dot(theta, np.pi), np.float32(180.0))

def to_degrees(theta):
    return np.divide(np.dot(theta, np.float32(180.0)), np.pi)

E_RADIUS = 6371e3
class Location:
    def __init__(self, lat, lon, alt=0):
        self.lat = lat
        self.lon = lon
        self.alt = alt

        self.FLOAT_LAT = lat
        self.FLOAT_LONG = lon
        if PGO:
            self.COORDS_LATITUDE = util.f2i(lat)
            self.COORDS_LONGITUDE = util.f2i(lon)
            self.COORDS_ALTITUDE = util.f2i(alt)

    def to_location_coords(self):
        return (self.COORDS_LATITUDE, self.COORDS_LONGITUDE, self.COORDS_ALTITUDE)

    def to_cell_id(self, level=15):
        return CellId.from_lat_lng(LatLng.from_degrees(self.lat, self.lon)).parent(level)

    def get_s2_neighbors_consecutive(self, radius=10):
        origin = CellId.from_lat_lng(LatLng.from_degrees(self.lat, self.lon)).parent(15)
        walk = [origin.id()]
        right = origin.next()
        left = origin.prev()

        # Search around provided radius
        for i in range(radius):
            walk.append(right.id())
            walk.append(left.id())
            right = right.next()
            left = left.prev()

        # Return everything
        return sorted(walk)

    def get_s2_neighbors_edge(self, **kwargs):
        # src: https://www.reddit.com/r/pokemongodev/comments/4tgfqz/how_does_the_server_respond_to_cell_ids_in_the/d5mgx04

        origin = CellId.from_lat_lng(LatLng.from_degrees(self.lat, self.lon)).parent(15)

        level = 15
        max_size = 1 << 30
        size = origin.get_size_ij(level)

        face, i, j = origin.to_face_ij_orientation()[0:3]

        walk = [origin.id(),
                origin.from_face_ij_same(face, i, j - size, j - size >= 0).parent(level).id(),
                origin.from_face_ij_same(face, i, j + size, j + size < max_size).parent(level).id(),
                origin.from_face_ij_same(face, i - size, j, i - size >= 0).parent(level).id(),
                origin.from_face_ij_same(face, i + size, j, i + size < max_size).parent(level).id(),
                origin.from_face_ij_same(face, i - size, j - size, j - size >= 0 and i - size >=0).parent(level).id(),
                origin.from_face_ij_same(face, i + size, j - size, j - size >= 0 and i + size < max_size).parent(level).id(),
                origin.from_face_ij_same(face, i - size, j + size, j + size < max_size and i - size >=0).parent(level).id(),
                origin.from_face_ij_same(face, i + size, j + size, j + size < max_size and i + size < max_size).parent(level).id()]

        if kwargs.get("more_neighbors", False):
            walk += [
                origin.from_face_ij_same(face, i, j - 2*size, j - 2*size >= 0).parent(level).id(),
                origin.from_face_ij_same(face, i - size, j - 2*size, j - 2*size >= 0 and i - size >=0).parent(level).id(),
                origin.from_face_ij_same(face, i + size, j - 2*size, j - 2*size >= 0 and i + size < max_size).parent(level).id(),
                origin.from_face_ij_same(face, i, j + 2*size, j + 2*size < max_size).parent(level).id(),
                origin.from_face_ij_same(face, i - size, j + 2*size, j + 2*size < max_size and i - size >=0).parent(level).id(),
                origin.from_face_ij_same(face, i + size, j + 2*size, j + 2*size < max_size and i + size < max_size).parent(level).id(),
                origin.from_face_ij_same(face, i + 2*size, j, i + 2*size < max_size).parent(level).id(),
                origin.from_face_ij_same(face, i + 2*size, j - size, j - size >= 0 and i + 2*size < max_size).parent(level).id(),
                origin.from_face_ij_same(face, i + 2*size, j + size, j + size < max_size and i + 2*size < max_size).parent(level).id(),
                origin.from_face_ij_same(face, i - 2*size, j, i - 2*size >= 0).parent(level).id(),
                origin.from_face_ij_same(face, i - 2*size, j - size, j - size >= 0 and i - 2*size >=0).parent(level).id(),
                origin.from_face_ij_same(face, i - 2*size, j + size, j + size < max_size and i - 2*size >=0).parent(level).id()]

        return walk

    def displace(self, theta, distance):
        """
        Displace a Location theta degrees counterclockwise and some
        meters in that direction.
        Notes:
            http://www.movable-type.co.uk/scripts/latlong.html
            0 DEGREES IS THE VERTICAL Y AXIS! IMPORTANT!
        Args:
            theta:    A number in degrees.
            distance: A number in meters.
        Returns:
            A new Location.
        """
        # src: https://gis.stackexchange.com/a/153719

        theta = np.float32(theta)

        delta = np.divide(np.float32(distance), np.float32(E_RADIUS))

        theta = to_radians(theta)
        lat1 = to_radians(self.lat)
        lng1 = to_radians(self.lon)

        lat2 = np.arcsin( np.sin(lat1) * np.cos(delta) +
                        np.cos(lat1) * np.sin(delta) * np.cos(theta) )

        lng2 = lng1 + np.arctan2( np.sin(theta) * np.sin(delta) * np.cos(lat1),
                                np.cos(delta) - np.sin(lat1) * np.sin(lat2))

        lng2 = (lng2 + 3 * np.pi) % (2 * np.pi) - np.pi

        return Location(to_degrees(lat2), to_degrees(lng2))

    def hexagon_neighbors(self, **kwargs):
        aoff = kwargs.get("angle_offset", 30)
        # distance from center over corner to center of next hexagon
        dist_corner = float(kwargs.get("distance_corner", 200))
        # distance over border to center of next hexagon
        dist_border = np.cos(to_radians(30)) * dist_corner

        return [self.displace(aoff+angle*60, dist_border) for angle in range(6)]

    def distance_to(self, loc, **kwargs):
        # src: http://gis.stackexchange.com/q/163785

        lat2 = loc.lat
        lon2 = loc.lon
        if not kwargs.get("west_east", True):
            lon2 = self.lon
        if not kwargs.get("north_south", True):
            lat2 = self.lat

        # phi = 90 - latitude
        phi1 = to_radians(90.0 - self.lat)
        phi2 = to_radians(90.0 - lat2)

        # theta = longitude
        theta1 = to_radians(self.lon)
        theta2 = to_radians(lon2)

        # Compute the spherical distance from spherical coordinates.
        # For two locations in spherical coordinates:
        # (1, theta, phi) and (1, theta', phi')cosine( arc length ) =
        # sin phi sin phi' cos(theta-theta') + cos phi cos phi' distance = rho * arc    length

        cos = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) +
            np.cos(phi1)*np.cos(phi2))
        arc = np.arccos(cos)*E_RADIUS

        return arc

    def __repr__(self):
        return "(%s,%s)" % (self.lat, self.lon)

    def __eq__(self, other):
        if not isinstance(other, Location):
            return ValueError

        return self.to_cell_id(level=16).__eq__(other.to_cell_id(level=16))

    def __lt__(self, other):
        if not isinstance(other, Location):
            return ValueError

        return self.to_cell_id(level=16).__lt__(other.to_cell_id(level=16))

def hexagon_rect(loc_nw, loc_se, **kwargs):
    log.debug("hexagon_rect: (NW->SE) %s, %s" % (loc_nw, loc_se))

    # angle offset: make east-west move horizontal
    aoff = 30
    # distance from center over corner to center of next hexagon
    dist_corner = float(kwargs.get("distance_corner", 200))
    # distance over border to center of next hexagon
    dist_border = np.cos(to_radians(30)) * dist_corner
    # length of hexagon border
    length_border = np.sin(to_radians(30)) * dist_corner
    # distance between rows
    dist_row = (dist_corner+length_border)/2.0
    log.debug("hexagon_rect: distance corner %f m, border %f m" % (dist_corner, dist_border))

    total_south = loc_nw.distance_to(loc_se, west_east=False, north_south=True)
    total_east = loc_nw.distance_to(loc_se, west_east=True, north_south=False)
    log.debug("hexagon_rect: distance south %f m, east %f m" % (total_south, total_east))

    locs = []
    # go south
    for south_step in range(math.ceil(total_south/dist_row) + 1):
        east_steps = math.ceil(total_east/dist_border)

        # every second west-east row is offset by half a hexagon
        if (south_step % 2) == 1:
            # go to south-east hexagon from start
            south_base = loc_nw.displace(aoff+2*60, dist_border)

            # go down the rows, subtract 1 row because we already went one row down
            south_base = south_base.displace(180, dist_row*(south_step-1))
        else:
            # just go down the rows
            south_base = loc_nw.displace(180, dist_row*south_step)

        # go east
        for east_step in range(east_steps):
            loc = south_base.displace(90, dist_border*east_step)
            locs.append(loc)

    return locs

def cell_rect(**k):
    r = RegionCoverer()
    level = k.get("level", 25)
    r.min_level = k.get("min_level", level)
    r.max_level = k.get("max_level", level)
    if k.get("max_cells", None):
        r.max_cells = k.get("max_cells")
    p1 = LatLng.from_degrees(k["lat1"], k["lon1"])
    p2 = LatLng.from_degrees(k["lat2"], k["lon2"])
    cell_ids = r.get_covering(LatLngRect.from_point_pair(p1, p2))

    return cell_ids

if __name__ == "__main__":
    loc_a = Location(52.519957, 13.383781)
    loc_b = Location(52.499068, 13.417307)

    # circular area demo
    print("Hexagon neighbors of: %s" % loc_a)
    print("->")
    neighborhood = [loc_a] + loc_a.hexagon_neighbors()
    print(neighborhood)
    
    print("\n---")

    # rectangular area demo
    print("Corner locations for rectangle %s" % ([loc_a,loc_b]))
    print("->")
    area = hexagon_rect(loc_a, loc_b)
    print(area)

    print("\n---")

    # s2 neighbor demo
    loc = area[23]
    print("S2 cell edge neighbors for %s" % loc)
    print("->")
    parent = loc.to_cell_id()
    print("Level 15 cell id: %s" % parent.id())
    cellids = loc.get_s2_neighbors_edge()
    print("Neighborhood (edges):")
    print(cellids)

    # s2 neighbors by id demo
    cellids = loc.get_s2_neighbors_consecutive()
    print("Neighborhood (ids):")
    print(cellids)

