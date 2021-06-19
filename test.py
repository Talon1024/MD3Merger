import md3merger
import struct
from collections import namedtuple
from ctypes import cdll, c_float, c_short, c_ubyte

if __name__ != "__main__":
    exit(0)

Cartesian = namedtuple("Cartesian", "x y z")

# Argument types for NormalToLatLong
normal_type = c_float * 3
outbytes_type = c_ubyte * 2

nativeLib = cdll.LoadLibrary("./q3normals.so")

nativeLib.NormalToLatLong.argtypes = [normal_type, outbytes_type]
nativeLib.LatLongToNormal.argtypes = [c_short, normal_type]

class Normal:
    def __init__(self, x=0, y=0, z=1):
        self.normal = Cartesian(x, y, z)

    def __iter__(self):
        return iter(self.normal)

    @staticmethod
    def normalize(normal=(0,0,1)):
        from math import sqrt
        length = sqrt(
            normal.x * normal.x +
            normal.y * normal.y +
            normal.z * normal.z
        )
        normalized = Cartesian(
            normal.x / length,
            normal.y / length,
            normal.z / length
        )
        return normalized

    def _get_normal(self):
        vec = Normal.normalize(self.normal)
        return normal_type(*vec)

    _as_parameter_ = property(_get_normal)

normals = (
    Normal(1, 0, 0),
    Normal(0, 1, 0),
    Normal(0, 0, 1),
    Normal(0, 0, -1),
    Normal(12, 15, 5),
    Normal(7, 2, 0),
    Normal(-3, 2, -1),
    Normal(-3, 7, 0),
    Normal(3, -8, -3),
)

print(("{: ^10} " * 3 + "{: <12} " * 2).format(
    "x", "y", "z", "outbytes(n)", "outbytes(py)"), "match")

for normal in normals:
    normal_str = ("{: ^10.4f} " * 3).format(*normal)
    # Run the native code to convert normal to latitude/longitude bytes
    latlong = outbytes_type()  # Modified by NormalToLatLong
    nativeLib.NormalToLatLong(normal, latlong)
    latlong_str = "{: <12}".format(repr(bytes(latlong)))
    # Run the Python code to convert normal to latitude/longitude bytes
    mylatlong = md3merger.MD3Normal.encode(normal.normal, False)
    mylatlong_str = "{: <12}".format(repr(bytes(mylatlong)))
    print(normal_str, latlong_str, mylatlong_str,
          mylatlong_str == latlong_str)

print(("{: <7} " + "{: ^7} " * 6 + "{: ^7}").format(
    "latlong", "x", "y", "z", "myx", "myy", "myz", "match"))

for latlong in range(-32768, 32769, 8):
    if latlong >= 32768: latlong = 32767
    latlong = c_short(latlong)
    normal = normal_type()
    # Run the native code to convert latitude/longitude to normal
    nativeLib.LatLongToNormal(latlong, normal)
    latlong_str = "{: <7}".format(latlong.value)
    # Run the Python code to convert latitude/longitude to normal
    mynormal = md3merger.MD3Normal.decode(latlong.value, False)
    normal_str = ("{: 7.4f} " * 3).format(*normal)
    normal_str = normal_str.replace("-0.0000", " 0.0000")
    mynormal_str = ("{: 7.4f} " * 3).format(*mynormal)
    mynormal_str = mynormal_str.replace("-0.0000", " 0.0000")
    match = normal_str == mynormal_str
    print(latlong_str, normal_str, mynormal_str, match)
