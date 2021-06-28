#!/usr/bin/env python3
"""Utilities for reading and writing Quake 3 MD3 models, and merging them
together.

Classes:
Matrix - Used for creating matrices and doing matrix multiplication with them
Transform - Generalization of transformations to apply, and building matrices
    for those transformations
MD3Model - Represents an MD3 model header/container
MD3Frame - Represents information about a frame for an MD3 model
MD3Tag - Represents a tag (attachment position) for an MD3 model
MD3Surface - Represents geometry data for an MD3 model
MergedModel - Allows more than one MD3 model to be merged together

Functions:
md3_string - Convert a bytes object to a fixed-length bytes object
unmd3_string - Convert a fixed-length string to a bytes object

Named tuples:
Cartesian - Represents X, Y, and Z coordinates in 3D right-handed Cartesian
    space
MD3Triangle - Container for MD3 triangle data (three vertex indices)
MD3Texcoord - Container for MD3 texture coordinate data (two floats)
MD3Vertex - Container for MD3 vertex data (position and normal)

Constants:
MAX_QPATH - The maximum number of characters in a shader path or name
MD3_XYZ_SCALE - The conversion factor from Cartesian to MD3 coordinates
"""
# Merge two or more MD3 models together
import argparse
import struct
import io
from collections import namedtuple
from array import array
from math import atan2, acos, cos, floor, sin, pi, sqrt, radians

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
    """Utilities to encode and decode normal vectors for MD3 vertices

    Static methods:
    encode - Convert a normal vector to two-byte MD3 normal format
    encode_number - Convert a normal vector to a 16-bit MD3 normal integer
    decode - Convert an MD3 normal integer to a normal vector
    decode_euler - Convert a latitude/longitude to a normal vector
    """
    @staticmethod
    def encode(normal=(0, 0, 0), gzdoom=True):
        """Convert a normal vector to two-byte MD3 normal format

        if gzdoom is true, special straight up/down normal vectors are not
        modified"""
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
    def encode_number(normal=(0, 0, 0), gzdoom=True):
        "Convert a normal vector to a 16-bit MD3 normal integer"
        latlongbytes = MD3Normal.encode(normal, gzdoom)
        return struct.unpack("<h", latlongbytes)[0]

    @staticmethod
    def decode(latlong=0):
        "Convert an MD3 normal integer to a normal vector"
        latlongbytes = struct.pack("<h", latlong)
        lng, lat = struct.unpack("<2b", latlongbytes)
        return MD3Normal.decode_euler(lat, lng)

    @staticmethod
    def decode_euler(lat=0, lng=0):
        "Convert a latitude/longitude to a normal vector"
        lat *= pi / 128
        lng *= pi / 128
        normal = Cartesian(
            cos(lat) * sin(lng),
            sin(lat) * sin(lng),
            cos(lng)
        )
        return normal


class Matrix:
    """A matrix of numbers, which can be multiplied with other matrices using
    the @ operator, or multiplied by scalars using the * operator

    Properties:
    rows - The number of rows in this matrix
    columns - The number of columns in this matrix

    Methods:
    row - Get the vector for the given row
    column - Get the vector for the given column"""
    def __init__(self, rows=3, columns=3, elements=None):
        self.elements = []
        for row in range(rows):
            self.elements.append([0] * columns)
            # Identity matrix
            if rows == columns:
                self.elements[row][row] = 1
        if elements is not None:
            self.elements[0:len(elements)] = elements[:]

    def __matmul__(self, other):
        "Multiply this matrix by another matrix"
        if not isinstance(other, Matrix):
            return None
        if self.columns != other.rows:
            return None
        new_matrix = Matrix(self.rows, other.columns)
        for row in range(self.rows):
            for col in range(other.columns):
                # Multiply the row of the first by the column of the second
                row_vector = self.row(row)
                column_vector = other.column(col)
                new_matrix.elements[row][col] = (
                    sum(map(lambda a, b: a * b, row_vector, column_vector)))
        return new_matrix

    def __mul__(self, other):
        "Multiply this matrix by a scalar"
        new_matrix = Matrix(self.rows, self.columns)
        for row_index in range(self.rows):
            for column_index in range(self.columns):
                new_matrix.elements[row_index][column_index] = (
                    self.elements[row_index][column_index] * other
                )
        return new_matrix

    def __truediv__(self, other):
        "Divide this matrix by a scalar"
        return self.__mul__(1./other)

    def __getitem__(self, key):
        return self.elements[key]

    def __str__(self):
        number_strings = []
        for row in self.elements:
            for number in row:
                number_strings.append("{: .4f}".format(float(number)))
        max_number_length = max(map(len, number_strings))
        row_format_str = (
            "[" +
            "{{:^{}}}".format(max_number_length) * self.columns +
            "]\n"
        )
        out = ""
        for row_index in range(self.rows):
            number_index = row_index * self.columns
            row = number_strings[number_index : number_index + self.columns]
            out += row_format_str.format(*row)
        return out

    @property
    def rows(self):
        "Get the number of rows in this matrix"
        return len(self.elements)

    @property
    def columns(self):
        "Get the number of columns in this matrix"
        return len(self.elements[0])

    def row(self, row_index):
        "Get the vector for the given row in this matrix"
        return self.elements[row_index]

    def column(self, column_index):
        "Get the vector for the given column in this matrix"
        return [x[column_index] for x in self.elements]


class Transform:
    """A helper which can build matrices for transformations

    Methods:
    angle_matrix - Generate a rotation matrix for a rotation on the Z axis
    pitch_matrix - Generate a rotation matrix for a rotation on the X axis
    roll_matrix - Generate a rotation matrix for a rotation on the Y axis
    rotation_matrix - Generate a matrix for rotations on all axes
    scale_matrix - Generate a scale matrix for this transformation's scale
    """
    def __init__(self,
                 position=(0, 0, 0),
                 angle=0,
                 pitch=0,
                 roll=0,
                 scale=(1, 1, 1)):
        self.position = Cartesian(*position)
        self.angle = angle
        self.pitch = pitch
        self.roll = roll
        self.scale = Cartesian(*scale)

    def angle_matrix(self):
        "Generate a rotation matrix for a rotation on the Z axis"
        rotation = Matrix(3, 3)
        theta = radians(self.angle)
        rotation[0][0] = cos(theta)
        rotation[0][1] = -sin(theta)
        rotation[1][0] = sin(theta)
        rotation[1][1] = cos(theta)
        return rotation

    def pitch_matrix(self):
        "Generate a rotation matrix for a rotation on the X axis"
        rotation = Matrix(3, 3)
        theta = radians(self.pitch)
        rotation[1][1] = cos(theta)
        rotation[1][2] = -sin(theta)
        rotation[2][1] = sin(theta)
        rotation[2][2] = cos(theta)
        return rotation

    def roll_matrix(self):
        "Generate a rotation matrix for a rotation on the Y axis"
        rotation = Matrix(3, 3)
        theta = radians(self.roll)
        rotation[0][0] = cos(theta)
        rotation[0][2] = sin(theta)
        rotation[2][0] = -sin(theta)
        rotation[2][2] = cos(theta)
        return rotation

    def scale_matrix(self):
        "Generate a matrix for this transformation's scale on each axis"
        # Assign values to the diagonal
        scale = Matrix()
        for index, value in enumerate(self.scale):
            scale[index][index] = value
        return scale

    def rotation_matrix(self):
        "Generate a matrix for rotations on all axes"
        return self.angle_matrix() @ self.pitch_matrix() @ self.roll_matrix()


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
        self.preprocess()
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
        new_model = MD3Model(self.name.decode("utf-8"))
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

        model = MD3Model(name.decode("utf-8"))

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
                 origin=(0, 0, 0),
                 bounds_min=(0, 0, 0),
                 bounds_max=(0, 0, 0),
                 name=""):
        self.name = name.encode("utf-8")
        self.radius = radius
        self.origin = origin
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max

    def get_data(self):
        data = bytearray(struct.pack("<3f", *self.bounds_min))
        data += struct.pack("<3f", *self.bounds_max)
        data += struct.pack("<3f", *self.origin)
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

    @property
    def frames(self):
        # Texture coordinates and vertices are "parallel arrays" for the first
        # frame, so the number of frames can be calculated by dividing the
        # length of the texture coordinates by the length of the vertices
        tc_count = len(self.texcoords)
        vert_count = len(self.vertices)
        return vert_count // tc_count

    def get_data(self):
        self.preprocess()
        if not self.is_valid():
            return b""
        data = bytearray(b"IDP3" + md3_string(self.texture, MAX_QPATH))
        tri_count = len(self.triangles)
        tc_count = len(self.texcoords)
        # The first 0 represents the "flags", which seem to be unused in the
        # MD3 format. The 1 is the number of shaders, and I don't see why
        # there needs to be any more than 1. Also, tc_count is used instead of
        # vert_count because tc_count is the number of vertices for one frame
        data += struct.pack("<5i", 0, self.frames, 1, tc_count, tri_count)
        shader_data = (
            md3_string(self.texture, MAX_QPATH) +
            struct.pack("<i", 0))
        tri_data = b"".join([
            struct.pack("<3i", *tri) for tri in self.triangles
        ])
        vert_data = b"".join([
            struct.pack("<4h", *vert) for vert in self.vertices
        ])
        tc_data = b"".join([
            struct.pack("<2f", *tc) for tc in self.texcoords
        ])
        # Ensure data offset numbers point to AFTER the offset themselves
        shaders_offset = len(data) + struct.calcsize("<5i")
        tris_offset = shaders_offset + len(shader_data)
        tc_offset = tris_offset + len(tri_data)
        verts_offset = tc_offset + len(tc_data)
        end_offset = verts_offset + len(vert_data)
        data += struct.pack(
            "<5i", tris_offset, shaders_offset,
            tc_offset, verts_offset, end_offset)
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
        new_surface = MD3Surface(self.texture.decode("utf-8"))
        new_surface.triangles = self.triangles[:]
        new_surface.vertices = self.vertices[:]
        new_surface.texcoords = self.texcoords[:]
        return new_surface

    def preprocess(self):
        pass

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
                tris.append(MD3Triangle(*tri))
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
                tcs.append(MD3Texcoord(*tc))
                count -= 1
            return {
                "type": "texcoords",
                "texcoords": tcs
            }

        def read_verts(count):
            verts = []
            while count > 0:
                vertex = struct.unpack("<4h", stream.read(8))
                verts.append(MD3Vertex(*vertex))
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
    def __init__(self, max_frames=None, name=b""):
        super().__init__(name)
        self.max_frames = max_frames
        self.surface_frames = 0  # Number of frames each surface should have
        self.texture_surfaces = {}
        self.frame_names = {}
        self.surface_transforms = []

    def add_model(self, model, transform=None):
        for surface in model.surfaces:
            self.add_surface(surface, transform)
        self.tags += model.tags
        for index, frame in enumerate(model.frames):
            if frame.name != b"":
                self.frame_names[index] = frame.name.decode("utf-8")

    def fix_surface_animations(self, surface):
        # Ensure a surface has the required amount of frames
        surface_frames = surface.frames
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
        for index in range(len(surface.vertices)):
            newx = surface.vertices[index].x
            newy = surface.vertices[index].y
            newz = surface.vertices[index].z
            newn = surface.vertices[index].n
            # Apply rotation
            if transform is not None:
                scale_matrix = transform.scale_matrix()
                vertex_matrix = Matrix(1, 3, [[newx], [newy], [newz]])
                vertex_matrix = scale_matrix @ vertex_matrix
                normal_matrix = (
                    Matrix(1, 3, [[co] for co in MD3Normal.decode(newn)]))
                transform_matrix = transform.rotation_matrix()
                vertex_matrix = transform_matrix @ vertex_matrix
                normal_matrix = transform_matrix @ normal_matrix
                newx, newy, newz = vertex_matrix.column(0)
                newn = MD3Normal.encode_number(normal_matrix.column(0))
                # Apply position
                newx += transform.position.x * MD3_XYZ_SCALE
                newy += transform.position.y * MD3_XYZ_SCALE
                newz += transform.position.z * MD3_XYZ_SCALE
            # The transformation could have converted the XYZ values to floats
            newx, newy, newz = map(
                lambda x: int(floor(x)),
                (newx, newy, newz))
            # I think the vertex should be copied regardless
            surface.vertices[index] = MD3Vertex(newx, newy, newz, newn)

    def add_surface(self, surface, transform=None):
        surflist = self.texture_surfaces.setdefault(surface.texture, [])
        surflist.append(surface)
        self.surfaces.append(surface)
        self.surface_transforms.append(transform)
        if surface.frames > self.surface_frames:
            self.surface_frames = surface.frames
            if (self.max_frames is not None and
                    self.surface_frames > self.max_frames):
                self.surface_frames = self.max_frames

    def preprocess(self):
        # Pre-process surfaces: apply transformations, and fix the animations
        for surface, transform in zip(self.surfaces, self.surface_transforms):
            self.apply_transform(surface, transform)
            self.fix_surface_animations(surface)
        # Rebuild surfaces
        self.surfaces = []
        self.frames = []
        for texture, surflist in self.texture_surfaces.items():
            new_surface = MD3Surface(texture.decode())
            # Ensure triangles from each surface reference the correct vertex
            tri_add = 0
            # Add triangles and UVs from the surface - these are only for the
            # first frame.
            for surface in surflist:
                for tri in surface.triangles:
                    new_tri = (vertex + tri_add for vertex in tri)
                    new_tri = MD3Triangle(*new_tri)
                    new_surface.triangles.append(new_tri)
                tri_add += len(surface.texcoords)
                new_surface.texcoords += surface.texcoords
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
        # Rebuild frames
        for frame_num in range(self.surface_frames):
            x_coords = array("h")
            y_coords = array("h")
            z_coords = array("h")
            # There's probably a better way to do this, but I don't know it.
            for surface in self.surfaces:
                verts_per_frame = len(surface.texcoords)
                frame_verts = surface.vertices[
                    frame_num * verts_per_frame :
                    (frame_num + 1) * verts_per_frame
                ]
                for vertex in frame_verts:
                    x_coords.append(vertex.x)
                    y_coords.append(vertex.y)
                    z_coords.append(vertex.z)
            coords = (x_coords, y_coords, z_coords)
            bounds_min = [min(co) / MD3_XYZ_SCALE for co in coords]
            bounds_max = [max(co) / MD3_XYZ_SCALE for co in coords]
            frame_name = self.frame_names.setdefault(frame_num, "")
            radius = max(
                sqrt(sum(map(lambda a: a * a, bounds_min))),
                sqrt(sum(map(lambda a: a * a, bounds_max))),
            )
            frame = MD3Frame(
                radius, (0, 0, 0), bounds_min, bounds_max, frame_name)
            self.frames.append(frame)


if __name__ == "__main__":
    from operator import eq
    from itertools import starmap
    # Cache - re-use loaded models
    MD3Cache = {}

    def add_model(model_arg):
        model_filename = model_arg.filename
        if model_filename in MD3Cache:
            model = MD3Cache[model_filename]
        else:
            model = MD3Model.from_stream(open(model_filename, "rb"))
            MD3Cache[model_filename] = model
        if all(starmap(eq, zip(model_arg[1:], MODEL_ARGUMENT_DEFAULTS))):
            transform = None
        else:
            transform = Transform(
                (model_arg.x, model_arg.y, model_arg.z),
                model_arg.yaw, model_arg.pitch, model_arg.roll,
                (model_arg.sx, model_arg.sy, model_arg.sz)
            )
        return model.clone(), transform

    ParsedModelArgument = namedtuple(
        "ParsedModelArgument", "filename x y z yaw pitch roll sx sy sz")
    MODEL_ARGUMENT_DEFAULTS = (
        0, 0, 0,  # Position
        0, 0, 0,  # Orientation
        1, 1, 1)  # Scale


    def model_argument(argument):
        # argument.md3@x@y@z|y|p|r
        filename_length = 0
        coordinates = {
            "@": [],
            "|": [],
            "%": []
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
            coordinates[element_type].append(element[1:])
        filename = argument[0:filename_length]
        # Ensure each list in "coordinates" has at least 3 members
        for coordkey in coordinates:
            excess = 0
            if coordkey == "%":
                excess = 1
            coordinates[coordkey] += (
                [excess] * (3 - len(coordinates[coordkey]))
            )
        x, y, z = tuple(map(float, coordinates["@"]))
        yaw, pitch, roll = tuple(map(float, coordinates["|"]))
        sx, sy, sz = tuple(map(float, coordinates["%"]))
        return ParsedModelArgument(
            filename,
            x, y, z,
            yaw, pitch, roll,
            sx, sy, sz)


    parser = argparse.ArgumentParser(
        description="Merge multiple Quake 3 MD3 models")
    parser.add_argument("models", nargs="+", type=model_argument, help="""
    The input models. Each model is of the form:

    model.md3[@x][@y][@z][|a][|p][|r][%%sx][%%sy][%%sz]

    It's basically the filename, followed by a transformation specification,
    which is a bunch of numbers prefixed by either an at sign (@), a pipe
    character (|), or a percent sign (%%), and which represent position,
    orientation, and scale respectively. The numbers and the symbols can be
    given in any order.
    """)
    parser.add_argument("--frames", type=int, help="Maximum animation frames")
    parser.add_argument("out_model", help="The output MD3 file")
    parsed_args = parser.parse_args()

    in_models = map(add_model, parsed_args.models)

    out_model = MergedModel(parsed_args.frames, parsed_args.out_model)
    for in_model in in_models:
        out_model.add_model(in_model[0], in_model[1])
    out_filename = parsed_args.out_model
    with open(out_filename, "wb") as out_file:
        out_data = out_model.get_data()
        out_file.write(out_data)
