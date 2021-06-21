#!/usr/bin/env python3
# Merge two or more MD3 models together
import argparse
import struct
import io
from collections import namedtuple

# Constants for MD3
MAX_QPATH = 64
# I don't care about limits, and neither does GZDoom
# MD3_MAX_FRAMES = 1024
# MD3_MAX_TAGS = 16
# MD3_MAX_SURFACES = 32
# MD3_MAX_SHADERS = 256
# MD3_MAX_VERTS = 4096
# MD3_MAX_TRIANGLES = 8192
MD3_XYZ_SCALE = 64


class Vector2:
    @staticmethod
    def from_bytes(data):
        return struct.unpack("<2f", data)

    @staticmethod
    def to_bytes(x, y):
        return struct.pack("<2f", x, y)


class Vector3:
    @staticmethod
    def from_bytes(data):
        return struct.unpack("<3f", data)

    @staticmethod
    def to_bytes(x, y, z):
        return struct.pack("<3f", x, y, z)


# Different input/output format than Vector3
class MD3Vector:
    @staticmethod
    def from_bytes(data):
        ox, oy, oz = struct.unpack("<3h", data)
        return MD3Vector.from_raw(ox, oy, oz)

    @staticmethod
    def from_raw(ox, oy, oz):
        x, y, z = tuple(map(
            lambda x: x / MD3_XYZ_SCALE, (ox, oy, oz)
        ))
        return x, y, z

    @staticmethod
    def to_bytes(ox, oy, oz):
        x, y, z = tuple(map(
            lambda x: x * MD3_XYZ_SCALE, (ox, oy, oz)))
        return struct.pack("<3h", x, y, z)


def md3_string(data, length, terminate=False):
    "Given a bytes object, return a fixed-size bytes object"
    if terminate:
        length -= 1
    if len(data) > length:
        return data[0:length]
    count = length - len(data)
    return data + b"\0" * count

def unmd3_string(data):
    "Given a fixed-size bytes object, return a bytes object"
    null_pos = data.find(b"\0")
    if null_pos >= 0:
        data = data[0:null_pos]
    return data


Cartesian = namedtuple("Cartesian", "x y z")


class MD3Normal:
    @staticmethod
    def encode(normal=(0,0,0), gzdoom=True):
        from math import atan2, acos, pi, sqrt

        normal = Cartesian(*normal)

        # Normalize vector
        length = sqrt(
            normal.x * normal.x +
            normal.y * normal.y +
            normal.z * normal.z
        )
        normal = Cartesian(
            normal.x / length,
            normal.y / length,
            normal.z / length
        )

        lng = int(atan2(normal.y, normal.x) * 127.5 / pi) & 0xff
        lat = int(acos(normal.z) * 127.5 / pi) & 0xff

        if not gzdoom:
            if normal.x == 0 and normal.y == 0:
                if normal.z > 0:
                    lng, lat = 0, 0
                else:
                    lng, lat = 0, 128
        return struct.pack("<2B", lat, lng)

    @staticmethod
    def decode(latlong=0, gzdoom=False):
        latlongbytes = struct.pack("<h", latlong)
        lng, lat = struct.unpack("<2b", latlongbytes)
        return MD3Normal.decode_euler(lat, lng)

    @staticmethod
    def decode_euler(lat=0, lng=0, gzdoom=False):
        from math import cos, sin, pi
        lat *= pi / 128
        lng *= pi / 128
        normal = Cartesian(
            cos(lat) * sin(lng),
            sin(lat) * sin(lng),
            cos(lng)
        )
        return normal


class Transform:
    def __init__(self,
                 position=(0,0,0),
                 angle=0,
                 pitch=0,
                 roll=0):
        self.position = position
        self.angle = angle
        self.pitch = pitch
        self.roll = roll

    def to_matrix(self):
        pass


class MD3Model:
    def __init__(self, name=""):
        self.name = name.encode("utf-8")
        self.surfaces = []
        self.frames = []
        self.tags = []

    def get_data(self):
        data = bytearray(b"IDP3")  # Magic number/ID
        data += struct.pack("<i", 15)  # Version
        data += md3_string(self.name, MAX_QPATH)  # Name
        data += struct.pack("<i", 0)  # Flags
        data += struct.pack(
            "<iii",
            len(self.frames),  # of frames
            len(self.tags),  # of tags
            len(self.surfaces) # of surfaces
        )
        data += struct.pack("<i", 0)  # of skins
        # 11 32-bit little endian signed integers, and the name of the MD3
        frames_offset = 108
        # Frames and tags have fixed sizes
        # tags_offset = sum([x.get_size() for x in self.frames],
        #                   start=frames_offset)
        tags_offset = frames_offset + len(self.frames) * 56
        # surfaces_offset = sum([x.get_size() for x in self.tags],
        #                       start=tags_offset)
        surfaces_offset = tags_offset + len(self.tags) * 112
        eof_offset = (
            surfaces_offset + sum([x.get_size() for x in self.surfaces]))
        data += struct.pack(
            "<4i",
            frames_offset,
            tags_offset,
            surfaces_offset,
            eof_offset
        )
        self.preprocess()
        for frame in self.frames:
            data += frame.get_data()
        for tag in self.tags:
            data += tag.get_data()
        for surface in self.surfaces:
            data += surface.get_data()
        return bytes(data)

    def get_size(self):
        # 11 32-bit little endian signed integers, and the name of the MD3
        size = 108
        # size += sum([x.get_size() for x in self.frames])
        size += 56 * len(self.frames)  # Frames and tags have fixed sizes
        # size += sum([x.get_size() for x in self.tags])
        size += 112 * len(self.tags)
        size += sum([x.get_size() for x in self.surfaces])
        return size

    # So that this can be overridden by sub-classes
    def preprocess(self):
        pass
    
    def clone(self):
        new_model = MD3Model(self.name)
        new_model.surfaces = [s.clone() for s in self.surfaces]
        new_model.frames = self.frames[:]
        new_model.tags = self.tags[:]
        return new_model

    @staticmethod
    def from_stream(stream):
        md3_start = stream.tell()
        magic = stream.read(4)
        if magic != b"IDP3":
            return None
        stream.seek(stream.tell() + 4)  # Skip version number
        name = unmd3_string(stream.read(64))
        (num_frames, num_tags, num_surfaces, offset_frames, offset_tags,
         offset_surfaces) = (
             struct.unpack("<4x3i4x3i4x", stream.read(36)))

        def read_frames(count):
            frames = []
            while count > 0:
                frames.append(MD3Frame.from_stream(stream))
                count -= 1
            return {
                "type": "frames",
                "frames": frames
            }

        def read_tags(count):
            tags = []
            while count > 0:
                tags.append(MD3Tag.from_stream(stream))
                count -= 1
            return {
                "type": "tags",
                "tags": tags
            }

        def read_surfaces(count):
            surfaces = []
            while count > 0:
                surfaces.append(MD3Surface.from_stream(stream))
                count -= 1
            return {
                "type": "surfaces",
                "surfaces": surfaces
            }
        
        DataReadInfo = namedtuple("DataReadInfo", "offset function count")
        data_read_infos = (
            DataReadInfo(offset_frames, read_frames, num_frames),
            DataReadInfo(offset_tags, read_tags, num_tags),
            DataReadInfo(offset_surfaces, read_surfaces, num_surfaces),
        )

        model = MD3Model(name)

        for info in data_read_infos:
            stream.seek(md3_start + info.offset)
            data = info.function(info.count)
            data_type = data["type"]
            setattr(model, data_type, data[data_type])

        return model

    @staticmethod
    def from_bytes(data):
        with io.BytesIO(data) as stream:
            return MD3Model.from_stream(stream)


class MD3Frame:
    def __init__(self,
                 radius=0,
                 origin=(0,0,0),
                 bounds_min=(0,0,0),
                 bounds_max=(0,0,0),
                 name=""):
        self.name = name.encode("utf-8")
        self.radius = radius
        self.origin = origin
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max

    def get_data(self):
        data = bytearray(Vector3.to_bytes(*self.bounds_min))
        data += Vector3.to_bytes(*self.bounds_max)
        data += Vector3.to_bytes(*self.origin)
        data += struct.pack("<f", self.radius)
        data += md3_string(self.name, 16)
        return bytes(data)

    def get_size(self):
        return 56  # 4 * 10 + 16

    @staticmethod
    def from_stream(stream):
        bounds_min = struct.unpack("<3f", stream.read(12))
        bounds_max = struct.unpack("<3f", stream.read(12))
        origin = struct.unpack("<3f", stream.read(12))
        radius = struct.unpack("<f", stream.read(4))[0]
        name = unmd3_string(stream.read(16)).decode("utf-8")
        return MD3Frame(radius, origin, bounds_min, bounds_max, name)

    @staticmethod
    def from_bytes(data):
        with io.BytesIO(data) as stream:
            return MD3Frame.from_stream(stream)


class MD3Tag:
    def __init__(self,
                 position=(0,0,0),
                 axisa=(0,0,0),
                 axisb=(0,0,0),
                 axisc=(0,0,0),
                 name=""):
        self.name = name.encode("utf-8")
        self.position = position
        self.axes = (axisa, axisb, axisc)

    def get_data(self):
        data = bytearray(MAX_QPATH + 4 * 10)
        pos = MAX_QPATH
        data[0:pos] = md3_string(self.name, MAX_QPATH)
        for coordinate in self.position:
            data[pos:pos+4] = struct.pack("<f", coordinate)
            pos += 4
        for axis in self.axes:
            for coordinate in axis:
                data[pos:pos+4] = struct.pack("<f", coordinate)
                pos += 4
        return bytes(data)

    def get_size(self):
        return 112  # 4 * 12 + MAX_QPATH
    
    @staticmethod
    def from_stream(stream):
        name = unmd3_string(stream.read(MAX_QPATH)).decode("utf-8")
        position = struct.unpack("<3f", stream.read(12))
        axes = (struct.unpack("<3f", stream.read(12)) for axis in range(3))
        return MD3Tag(position, *axes, name)

    @staticmethod
    def from_bytes(data):
        with io.BytesIO(data) as stream:
            return MD3Tag.from_stream(stream)


MD3Triangle = namedtuple("MD3Triangle", "a b c")
MD3Texcoord = namedtuple("MD3Texcoord", "s t")
MD3Vertex = namedtuple("MD3Vertex", "x y z n")


class MD3Surface:
    def __init__(self, texture=""):
        self.texture = texture.encode("utf-8")
        self.triangles = []
        self.vertices = []
        self.texcoords = []

    def is_valid(self):
        base_vert_count = len(self.texcoords)
        vert_count = len(self.vertices)
        return vert_count % base_vert_count == 0

    def get_frames(self):
        # Texture coordinates and vertices are "parallel arrays" for the first
        # frame, so the number of frames can be calculated by dividing the
        # length of the texture coordinates by the length of the vertices
        tc_count = len(self.texcoords)
        vert_count = len(self.vertices)
        return vert_count // tc_count

    def get_data(self):
        if not self.is_valid():
            return b""
        data = bytearray(b"IDP3" + md3_string(self.texture, MAX_QPATH))
        tri_count = len(self.triangles)
        tc_count = len(self.texcoords)
        frame_count = self.get_frames()
        # The first "0" represents the "flags", which seem to be unused in the
        # MD3 format. The "1" is the number of surfaces, and I don't see why
        # there needs to be any more than 1. Also, tc_count is used instead of
        # vert_count because tc_count is the number of vertices for one frame
        data += struct.pack("<5i", 0, frame_count, 1, tc_count, tri_count)
        shaders_offset = len(data)
        shader_data = (
            md3_string(self.texture, MAX_QPATH) +
            struct.pack("<i", 0))
        tri_data = b"".join([
            struct.pack("<3i", tri) for tri in self.triangles
        ])
        vert_data = b"".join([
            struct.pack("<4h", vert) for vert in self.vertices
        ])
        tc_data = b"".join([
            struct.pack("<2f", tc) for tc in self.texcoords
        ])
        tris_offset = shaders_offset + len(shader_data)
        tc_offset = tris_offset + len(tri_data)
        verts_offset = tc_offset + len(tc_data)
        end_offset = verts_offset + len(vert_data)
        data += struct.pack("<5i",
            tris_offset, shaders_offset, tc_offset, verts_offset, end_offset)
        data += shader_data
        data += tri_data
        data += tc_data
        data += vert_data
        return bytes(data)

    def get_size(self):
        tri_count = len(self.triangles)
        vert_count = len(self.vertices)
        tc_count = len(self.texcoords)
        size = 4 * 12 + MAX_QPATH  # ID, name, flags, counts, and offsets
        size += 4 + MAX_QPATH  # Shader objects - only one is used
        size += tri_count * 12  # Triangles - 3 vertex indices each
        size += tc_count * 8  # Texture coordinates - 2 floats each
        size += vert_count * 8  # Vertex positions - 4 shorts each
        return size
    
    def clone(self):
        new_surface = MD3Surface(self.texture.decode())
        new_surface.triangles = self.triangles[:]
        new_surface.vertices = self.vertices[:]
        new_surface.texcoords = self.texcoords[:]
        return new_surface

    @staticmethod
    def from_stream(stream):
        surface_start = stream.tell()
        magic_id = stream.read(4)
        if magic_id != b"IDP3":
            return  # Not a valid surface
        stream.seek(stream.tell() + 4 + MAX_QPATH)  # Skip flags and name
        frame_count, shader_count, vert_count, tri_count = (
            struct.unpack("<4i", stream.read(16))
        )
        offset_tris, offset_shaders, offset_tcs, offset_verts = (
            struct.unpack("<4i4x", stream.read(20))
        )

        def read_shaders(count):
            # Only read the first shader
            shader = unmd3_string(stream.read(MAX_QPATH))
            stream.seek(stream.tell() + 4)  # Skip shader index
            # Skip the other shaders
            while count > 0:
                stream.seek(stream.tell() + MAX_QPATH + 4)
                count -= 1
            return {
                "type": "texture",
                "texture": shader
            }

        def read_tris(count):
            tris = []
            while count > 0:
                tri = struct.unpack("<3i", stream.read(12))
                tris.append(MD3Triangle(tri))
                count -= 1
            return {
                "type": "triangles",
                "triangles": tris
            }

        def read_tcs(count):
            # "count" here is the same as the number of vertices
            tcs = []
            while count > 0:
                tc = struct.unpack("<2f", stream.read(8))
                tcs.append(MD3Texcoord(tc))
                count -= 1
            return {
                "type": "texcoords",
                "texcoords": tcs
            }

        def read_verts(count):
            verts = []
            while count > 0:
                vertex = struct.unpack("<4h", stream.read(8))
                verts.append(MD3Vertex(vertex))
                count -= 1
            return {
                "type": "vertices",
                "vertices": verts
            }

        DataReadInfo = namedtuple("DataReadInfo", "offset function count")

        data_read_infos = (
            DataReadInfo(offset_shaders, read_shaders, shader_count),
            DataReadInfo(offset_tris, read_tris, tri_count),
            DataReadInfo(offset_tcs, read_tcs, vert_count),
            DataReadInfo(offset_verts, read_verts, vert_count * frame_count),
        )

        surface = MD3Surface()

        for info in data_read_infos:
            stream.seek(surface_start + info.offset)
            data = info.function(info.count)
            data_type = data["type"]
            data_value = data[data_type]
            setattr(surface, data_type, data_value)
        return surface
    
    @staticmethod
    def from_bytes(data):
        with io.BytesIO(data) as stream:
            return MD3Surface.from_stream(stream)



class MergedModel(MD3Model):
    def __init__(self, max_frames=-1, name=b""):
        super().__init__(name)
        self.max_frames = max_frames
        self.surface_frames = 0  # Number of frames each surface should have
        self.texture_surfaces = {}
        self.surface_transforms = []

    def add_model(self, model, transform=None):
        for surface in model.surfaces:
            self.add_surface(surface, transform)
        self.tags += model.tags

    def fix_surface_animations(self, surface):
        # Ensure a surface has the required amount of frames
        surface_frames = surface.get_frames()
        if surface_frames != self.surface_frames:
            # Surface has more frames, so truncate the animation
            if surface_frames > self.surface_frames:
                max_vertex = len(surface.texcoords) * self.surface_frames
                surface.vertices = surface.vertices[0:max_vertex]
            # Surface has fewer frames, so extend animation from last frame
            else:
                missing_frames = self.surface_frames - surface_frames
                last_frame_verts = surface.vertices[
                    (surface_frames - 1) * len(surface.texcoords):-1]
                surface.vertices += last_frame_verts * missing_frames

    def apply_transform(self, surface, transform):
        # Move all of a surface's vertices based on a transformation matrix
        pass

    def add_surface(self, surface, transform=None):
        if surface.texture not in self.texture_surfaces:
            surflist = self.texture_surfaces.setdefault(surface.texture, [])
            surflist.append(surface)
        self.surfaces.append(surface)
        self.surface_transforms.append(transform)

    def preprocess(self):
        # Pre-process surfaces: apply transformations, and fix the animations
        for surface, transform in zip(self.surfaces, self.surface_transforms):
            self.apply_transform(surface, transform)
            self.fix_surface_animations(surface)
        # Rebuild surfaces and frames
        self.surfaces = []
        self.frames = []
        for texture, surflist in self.texture_surfaces.items():
            new_surface = MD3Surface(texture)
            # Ensure triangles from each surface reference the correct vertex
            tri_add = 0
            # Add triangles and UVs from the surface - these are only for the
            # first frame.
            for surface in surflist:
                new_surface.texcoords += surface.texcoords
                for tri in surface.triangles:
                    new_tri = (vertex + tri_add for vertex in tri)
                    new_surface.triangles += new_tri
                tri_add += len(surface.triangles)
            # Add vertices from each frame in each surface
            for frame in range(self.surface_frames):
                for surface in surflist:
                    verts_per_frame = len(surface.texcoords)
                    new_surface.vertices += (
                        surface.vertices[
                            frame * verts_per_frame :
                            (frame + 1) * verts_per_frame
                        ])
            self.surfaces.append(new_surface)


if __name__ == "__main__":
    # Cache - re-use loaded models
    MD3Cache = {}

    def add_model(model_filename):
        if model_filename in MD3Cache:
            model = MD3Cache[model_filename]
        else:
            model = MD3Model.from_stream(open(model_filename, "rb"))
            MD3Cache[model_filename] = model
        return model

    ParsedModelArgument = namedtuple(
        "ParsedModelArgument", "filename x y z yaw pitch roll")


    class MyQueue:
        def __init__(self, data=[], index=0):
            self.data = data
            self.index = index

        def add(self, element):
            if self.index < len(self.data):
                self.data[self.index] = element
            else:
                self.data.append(element)
            self.index += 1


    def model_argument(argument):
        # argument.md3@x@y@z|y|p|r
        filename_length = 0
        coordinates = {
            "@": MyQueue(["0", "0", "0"]),
            "|": MyQueue(["0", "0", "0"])
        }
        subarguments = []
        coordinate_start = 0
        # Parse position
        pos = 0
        # Collect sub-arguments (position, orientation)
        while pos < len(argument):
            if argument[pos] in coordinates:
                if filename_length == 0:
                    filename_length = pos
                if coordinate_start > 0:
                    subarguments.append(argument[coordinate_start:pos])
                coordinate_start = pos
            pos += 1
        if filename_length == 0:
            filename_length = pos
        if coordinate_start > 0:
            subarguments.append(argument[coordinate_start:pos])
        # Parse sub-arguments
        for element in subarguments:
            element_type = element[0]
            coordinates[element_type].add(element[1:])
        filename = argument[0:filename_length]
        x, y, z = tuple(map(float, coordinates["@"].data))
        yaw, pitch, roll = tuple(map(float, coordinates["|"].data))
        return ParsedModelArgument(filename, x, y, z, yaw, pitch, roll)


    parser = argparse.ArgumentParser(
        description="Merge multiple Quake 3 MD3 models")
    parser.add_argument("models", nargs="+", type=model_argument)
    parser.add_argument("--frames", type=int, help="Maximum animation frames")
    parser.add_argument("out_model", help="The output MD3 file")
    parsed_args = parser.parse_args()

    models = map(add_model, parsed_args.models)

    out_model = MergedModel(parsed_args.frames, parsed_args.out_model)
