import argparse
import itertools
import os
import struct
from collections import defaultdict
from pprint import pprint

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

# Read functions

def read_vector(file, num=3):
    if num == 2:
        return Gf.Vec2f(*read_floats(file, num))
    if num == 3:
        return Gf.Vec3f(*read_floats(file, num))
    elif num == 4:
        return Gf.Vec4f(*read_floats(file, num))
    else:
        raise ValueError(f"Invalid number of vector components ({num})!")

def read_string(file):
    string_size = struct.unpack('<I', file.read(4))[0]
    return file.read(string_size).decode('ascii')

def read_strings(file, num):
    return tuple(read_string(file) for _ in range(num))

def read_float(file):
    return _read_single(file, "f")

def read_floats(file, num):
    return _read_multiple(file, "f", num)

def read_uint(file):
    return _read_single(file, "I")

def read_uints(file, num):
    return _read_multiple(file, "I", num)

def read_ushorts(file, num):
    return _read_multiple(file, "H", num)

def read_uchars(file, num):
    return _read_multiple(file, "B", num)

def _read_single(file, format):
    size = {
        "f": 4,
        "I": 4,
        "H": 2,
        "B": 1
    }[format]
    return struct.unpack("<" + format, file.read(size))[0]

def _read_multiple(file, format, num):
    return tuple(_read_single(file, format) for _ in range(num))


# Write functions

def write_vector(file, value):
    write_floats(file, value)

def write_string(file, string):
    write_uint(file, len(string))
    file.write(string.encode("ascii"))

def write_uint(file, value):
    _write_single(file, "I", value)

def write_uints(file, values):
    _write_multi(file, "I", values)

def write_float(file, value):
    _write_single(file, "f", value)

def write_floats(file, values):
    _write_multi(file, "f", values)

def write_ushorts(file, values):
    _write_multi(file, "H", values)

def write_uchar(file, value):
    _write_single(file, "B", value)

def write_uchars(file, values):
    _write_multi(file, "B", values)

def _write_single(file, format, value):
    file.write(struct.pack("<" + format, value))

def _write_multi(file, format, values):
    file.write(struct.pack("<" + (format * len(values)), *values))


# Data transformation functions

def flatten_array(indices, indexed_array):
    return [indexed_array[i] for i in indices]

def unflatten_array(flat_array, deduplicate=True):
    indexed_array = []
    indices = []
    for item in flat_array:
        if item in indexed_array and deduplicate:
            indices.append(indexed_array.index(item))
        else:
            indexed_array.append(item)
            indices.append(len(indexed_array) - 1)
    return indices, indexed_array

# Returns a dictionary mapping the flat array's unique items to the indices at which they appear
def inverse_index(flat_array):
    mapping = defaultdict(list)
    for i, item in enumerate(flat_array):
       mapping[item].append(i)
    return mapping

# Turn a tuple of arrays into an array of tuples, and vice versa
def transposed(*arrays, container_type=tuple):
    array_length = len(arrays[0])
    assert all(len(a) == array_length for a in arrays)
    array_slices = []
    for i in range(array_length):
        array_slices.append(container_type(a[i] for a in arrays))
    return array_slices

def to_tri_array(fvtx_array):
    tri_array = []
    for i in range(0, len(fvtx_array), 3):
        item = fvtx_array[i]
        assert item == fvtx_array[i+1] == fvtx_array[i+2]
        tri_array.append(item)
    return tri_array

def to_fvtx_array(tri_array):
    ret = []
    for item in tri_array:
        ret.extend([item] * 3)
    return ret


class Mesh:
    supported_types = "MSH", "ARE", "SHL"
    type_version_defaults = {
        "MSH": 1,
        "ARE": 1,
        "SHL": 6
    }

    def __init__(self, mesh_type, version, bounds, geometry, locators, hitboxes, textures_dir):
        self._mesh_type = mesh_type
        self._version = version
        self._bounds = bounds
        self.geometry = geometry
        self._locators = locators
        self._hitboxes = hitboxes
        self._textures_dir = textures_dir

        self.validate_attribs()
    
    def validate_attribs(self):
        mesh_attribs = {
            "MSH": {"position", "normal", "colour", "uv", "material"},
            "ARE": {"position", "normal", "colour", "uv", "material"},
            "SHL": {"position", "normal", "colour", "uv", "material", "piece_id",
                    "deform_position", "deform_normal",
                    "octant_id1", "octant_id2", "octant_weight1", "octant_weight2"}
        }
        our_attribs = set(self.geometry.attrib_names())
        missing = mesh_attribs[self._mesh_type].difference(our_attribs)
        assert not missing, f"Mesh is missing these attributes: {', '.join(missing)}"
        
        if self._mesh_type == "SHL":
            assert len(self._hitboxes) == 16
            assert "p" in self._locators, ("SHL (armour) meshes require a locator named \"p\" (an "
                                           "Empty object in Blender) to mark the origin.")
        elif not mesh_attribs["SHL"].difference(our_attribs):
            print("Mesh has all the required attributes for SHL (armour) type. "
                  "If you meant to use this type, run the same command with \"--mesh-type SHL\".")

    class UnsupportedMeshError(Exception):
        pass

    class Bounds:
        def __init__(self, mins, maxs, centre=None, radius=None):
            self.mins = Gf.Vec3f(mins)
            self.maxs = Gf.Vec3f(maxs)
            #assert all(mins[i] <= maxs[i] for i in range(3))
            if centre:
                self.centre = Gf.Vec3f(centre)
            else:
                self.centre = (self.mins + self.maxs) / 2
            if radius:
                self.radius = radius
            else:
                # TODO: this won't be accurate if the centre isn't really the centre
                self.radius = (self.maxs - self.centre).GetLength()

        def extent(self):
            return self.maxs - self.centre
        
        @classmethod
        def from_points(cls, points, padding=Gf.Vec3f(0., 0., 0.)):
            minx = min(v[0] for v in points)
            miny = min(v[1] for v in points)
            minz = min(v[2] for v in points)
            maxx = max(v[0] for v in points)
            maxy = max(v[1] for v in points)
            maxz = max(v[2] for v in points)
            mins = Gf.Vec3f(minx, miny, minz)
            maxs = Gf.Vec3f(maxx, maxy, maxz)
            return cls(mins - padding, maxs + padding)
        
        @classmethod
        def read_from_file(cls, f):
            return cls(read_vector(f), read_vector(f), read_vector(f), read_float(f))
        
        def write_to_file(self, f):
            write_vector(f, self.mins)
            write_vector(f, self.maxs)
            write_vector(f, self.centre)
            write_float(f, self.radius)

    class Geometry:
        """TODO: Write a description"""
        def __init__(self, **kwargs):
            # NOTE: If you're using Python 3.6 or earlier, `kwargs` isn't guaranteed to provide the
            # arguments in the correct order (though it probably will anyway)
            if not kwargs:
                raise ValueError("Can't make an empty Geometry")
            self._dict = kwargs
            self._keys = tuple(kwargs.keys())
            self.validate()

        def __len__(self):
            return len(self.as_attrib_tuple()[0])

        def validate(self):
            # Let's do some sanity checks
            # These *should* all have 1 item per face-vertex
            num = len(self)
            assert num % 3 == 0
            assert all(len(a) == num for a in self.as_attrib_tuple())

        def merge(self, other_geo):
            assert set(other_geo._keys) == set(self._keys)
            for key in self._keys:
                self._dict[key].extend(other_geo._dict[key])
            self.validate()
        
        class _GeoIterator:
            def __init__(self, parent_iterator, original_callable=None):
                self._parent_iterator = parent_iterator
                self._original_callable = original_callable
            
            # Originally this just returned `iter(self._parent_iterator)`, but it got annoying
            # having to make a new GeoIterator every time because the parent one got exhausted,
            # so instead let's just make a copy here.
            # Apparently this isn't very memory-efficient, but... who cares
            def __iter__(self):
                if self._original_callable:
                    ret = self._original_callable()
                    return ret
                iter1, iter2 = itertools.tee(self._parent_iterator)
                self._parent_iterator = iter1  # The original won't be valid anymore
                return iter2

            def __len__(self):
                if self._original_callable:
                    return len(list(self._original_callable()))
                iter1, iter2 = itertools.tee(self._parent_iterator)
                self._parent_iterator = iter1  # The original won't be valid anymore
                return len(list(iter2))
        
            @staticmethod
            def chainable(method):
                def wrapper(self, *args, **kwargs):
                    def func():
                        yield from method(self, *args, **kwargs)
                    return Mesh.Geometry._GeoIterator(func())
                return wrapper

            @chainable
            def unique_elements(self):
                seen = set()
                for element in self:
                    if element not in seen:
                        seen.add(element)
                        yield element
            
            @chainable
            def indices_in(self, unique_elements):
                unique_list = list(unique_elements)
                for element in self:
                    yield unique_list.index(element)

            @chainable
            def values_of(self, attrib_name):
                for element in self:
                    yield element.attr(attrib_name)
            
            @chainable
            def ordered_by(self, attrib_name):
                yield from sorted(self, key=lambda e: e.attr(attrib_name))
            
            @chainable
            def filtered_by(self, attrib_name, attrib_value):
                for element in self:
                    if element.attr(attrib_name) == attrib_value:
                        yield element
            
            @chainable
            def split_by(self, attrib_name):
                for unique_value in self.values_of(attrib_name).unique_elements():
                    yield unique_value, self.filtered_by(attrib_name, unique_value)

        class GeoElement(tuple):
            def __new__(cls, attrib_names, iterable):
                return super().__new__(cls, iterable)

            def __init__(self, attrib_names, iterable):
                self._attrib_names = attrib_names

            def attr(self, key):
                return self[self._attrib_names.index(key)]
        
        def attribute(self, attrib_name):
            return self._dict[attrib_name]
        
        def has_attribute(self, attrib_name):
            return attrib_name in self._dict
        
        def element(self, index):
            element_tuple = tuple(self.attribute(key)[index] for key in self._keys)
            return Mesh.Geometry.GeoElement(self._keys, element_tuple)
        
        def as_attrib_tuple(self):
            return tuple(self._dict[key] for key in self._keys)
        
        def attrib_names(self):
            return self._keys
        
        def elements(self):
            return Mesh.Geometry._GeoIterator(None, lambda: (self.element(i) for i in range(len(self))))
        
        def iterate_from(self, iterable):
            return Mesh.Geometry._GeoIterator(None, lambda: (i for i in iterable))

    def write_to_rwm(self, f):
        print("Writing to RWM".center(80, "="))

        elements = self.geometry.elements()
        materials = elements.values_of("material")
        ordered_materials = sorted(set(materials))

        f.write(self._mesh_type.encode("ascii"))
        write_uchar(f, self._version)
        write_uint(f, len(self._locators))
        write_uint(f, len(ordered_materials))
        write_uint(f, len(self.geometry.elements().unique_elements())) # Number of vertices
        write_uint(f, len(self.geometry) // 3)  # Number of triangles
        write_uint(f, len(self.geometry.elements().values_of("piece_id").unique_elements()))  # Number of pieces

        self._bounds.write_to_file(f)

        for loc_name, matrix in self._locators.items():
            write_string(f, loc_name)
            assert len(matrix) == 4
            for vec in matrix:
                assert len(vec) == 4
                write_floats(f, vec)
            write_uint(f, 0)  # Also dunno, flags?

        for material in ordered_materials:
            mat_name, spec_map, mat_colour = material
            assert type(spec_map) is str
            write_string(f, mat_name)
            write_string(f, spec_map)
            assert(len(mat_colour) == 4)
            write_uchars(f, tuple(int(c*255) for c in mat_colour))
            write_float(f, 30.)  # ???
            write_uint(f, 0)  # Also dunno, flags?

        face_vertices = self.geometry.elements().ordered_by("piece_id").ordered_by("material")
        vertices = face_vertices.unique_elements()

        for material, mat_facevertices in face_vertices.split_by("material"):
            write_uint(f, 0)  # Flags
            mat_vertices = mat_facevertices.unique_elements()
            write_uint(f, len(mat_vertices))
            for vertex in mat_vertices:
                write_floats(f, vertex.attr("position"))
                write_floats(f, vertex.attr("normal"))
                colour = vertex.attr("colour")
                assert len(colour) == 4
                write_uchars(f, tuple(int(c*255) for c in colour))
                uv = vertex.attr("uv")
                write_floats(f, (uv[0], -uv[1]))  # Flip V
            
            mat_piece_facevertices = mat_facevertices.split_by("piece_id")
            write_uint(f, len(mat_piece_facevertices))  # Number of pieces for this material
            for piece_id, facevertices_to_write in mat_piece_facevertices:
                write_uint(f, piece_id)
                num_fvtxs = len(facevertices_to_write)
                assert num_fvtxs % 3 == 0
                write_uint(f, num_fvtxs // 3)
                indices = facevertices_to_write.indices_in(vertices)
                write_ushorts(f, indices)
        
        if self._mesh_type == "SHL":
            write_uint(f, len(vertices))
            for v in vertices:
                write_floats(f, v.attr("deform_position"))
                write_floats(f, v.attr("deform_normal"))

            num_octants = 8
            write_uint(f, num_octants)
            write_uint(f, 2)  # Number of weights per vertex
            for v in vertices:
                write_uint(f, v.attr("octant_id1"))
                write_float(f, v.attr("octant_weight1"))
                write_uint(f, v.attr("octant_id2"))
                write_float(f, v.attr("octant_weight2"))
            for hitbox in self._hitboxes:
                hitbox.write_to_file(f)
            
        if self._mesh_type in ("SHL", "ARE"):
            write_uint(f, 0)  # Number of sprite types

    @classmethod
    def read_from_rwm(cls, f):
        print("Reading from RWM".center(80, "="))
        header = f.read(4)
        mesh_type = header[:3].decode("ascii")
        version = int(header[-1])
        print(f"{mesh_type} mesh type, version {version}")
        if not mesh_type in cls.supported_types:
            raise Mesh.UnsupportedMeshError(f"Unsupported mesh type \"{mesh_type}\"! "
                                            f"Currently, only {', '.join(cls.supported_types)} "
                                            "are supported.")
        num_locators = read_uint(f)
        print(f"Number of locators: {num_locators}")
        num_materials = read_uint(f)
        print(f"Number of materials: {num_materials}")
        total_num_verts = read_uint(f)
        print(f"Total number of verts: {total_num_verts}")
        total_num_tris = read_uint(f)
        print(f"Total number of triangles: {total_num_tris}")
        if not (total_num_verts and total_num_tris):
            raise Mesh.UnsupportedMeshError("No geometry in this mesh!")
        total_num_pieces = read_uint(f)
        print(f"Total number of mesh pieces: {total_num_pieces}")
        if mesh_type == "SHL":
            assert total_num_pieces == 16
        else:
            assert total_num_pieces == 1
        
        bounds = cls.Bounds.read_from_file(f)
        print(f"Bounding box min: {bounds.mins}")
        print(f"Bounding box max: {bounds.maxs}")
        print(f"Bounding box centre: {bounds.centre}")
        print(f"Bounding radius: {bounds.radius}")

        locators = {}
        for _ in range(num_locators):
            loc_name = read_string(f)
            print(f"\nLocator: {loc_name}")
            matrix = [list(read_floats(f, 4)) for _ in range(4)]
            pprint(matrix)
            locators[loc_name] = matrix
            assert read_uint(f) == 0

        materials = []
        for _ in range(num_materials):
            mat_name = read_string(f)
            if mat_name:
                print(f"\nMaterial name: {mat_name or '<no name>'}")
            spec_map = read_string(f)  # The specular map may be an empty string if not present
            print(f"Specular map: {spec_map or '<no spec map>'}")
            mat_colour = tuple(c / 255. for c in read_uchars(f, 4))
            print(f"Material colour(?): {mat_colour}")
            print(f"Specularity(?): {read_float(f)}")
            flags = read_uint(f)
            print(f"Flags: {flags}")
            if flags & 0x20:
                print(f"UV scrolling: {read_floats(f, 2)}")
            materials.append((mat_name, spec_map, mat_colour))

        fvtx_dict = defaultdict(list)
        vtx_dict = defaultdict(list)

        for material in materials:
            mesh_section_flags = read_uint(f)
            num_verts = read_uint(f)
            print(f"\nNumber of vertices for {material[0]}: {num_verts}")
            if mesh_section_flags != 0:
                print(f"{material[0]} is an rare one, because its flags are {hex(mesh_section_flags)}")

            for _ in range(num_verts):
                vtx_dict["position"].append(read_floats(f, 3))
                vtx_dict["normal"].append(read_floats(f, 3))
                vtx_dict["colour"].append(tuple(c / 255. for c in read_uchars(f, 4)))
                uv = read_floats(f, 2)
                vtx_dict["uv"].append((uv[0], -uv[1]))  # Flip V
                vtx_dict["material"].append(material)

            num_mat_pieces = read_uint(f)
            if mesh_type == "SHL":
                print(f"Number of pieces for {material[0]}: {num_mat_pieces}")
            else:
                assert num_mat_pieces == 1

            for _ in range(num_mat_pieces):
                piece_id = read_uint(f)
                num_tris = read_uint(f)
                print(f"Number of triangles for {material[0]} piece {piece_id}: {num_tris}")
                fvtx_dict["indices"].extend(read_ushorts(f, num_tris * 3))
                fvtx_dict["piece_id"].extend([piece_id] * num_tris * 3)

        assert all(idx < total_num_verts for idx in fvtx_dict["indices"])

        if mesh_type == "SHL":
            hitboxes = []
            num_deform_verts = read_uint(f)
            print(f"\nNumber of deformation vertices: {num_deform_verts}")
            assert num_deform_verts == total_num_verts
            for _ in range(num_deform_verts):
                vtx_dict["deform_position"].append(read_floats(f, 3))
                deform_normal = read_floats(f, 3)
                assert all(-1 <= x <= 1. for x in deform_normal)
                vtx_dict["deform_normal"].append(deform_normal)

            num_octants, num_weights = read_uints(f, 2)
            print(f"Number of armour octants: {num_octants}")
            print(f"Number of weights: {num_weights}")
            assert num_octants == 8
            assert num_weights == 2
            octant_set = set()
            for _ in range(num_deform_verts):
                octant1 = read_uint(f)
                octant_set.add(octant1)
                weight1 = read_float(f)
                octant2 = read_uint(f)
                octant_set.add(octant2)
                weight2 = read_float(f)
                assert 0. <= weight1 <= 1.
                assert 0. <= weight2 <= 1.
                assert abs(1 - (weight1 + weight2)) < 0.01  # They should add up to 1
                vtx_dict["octant_id1"].append(octant1)
                vtx_dict["octant_id2"].append(octant2)
                vtx_dict["octant_weight1"].append(weight1)
                vtx_dict["octant_weight2"].append(weight2)
            assert octant_set == set(range(num_octants))
            for _ in range(total_num_pieces):
                hitboxes.append(Mesh.Bounds.read_from_file(f))
        else:
            hitboxes = []

        if mesh_type in ("ARE", "SHL"):
            num_sprite_types = read_uint(f)
            print(f"\nNumber of sprite types: {num_sprite_types}")
            # Since I've only found one example of a sprite type, I'm not sure if they're ordered
            # as all sprite types followed by all sprites, or each array of sprites preceded by
            # their sprite type
            for _ in range(num_sprite_types):
                print(f"Sprite type name: {read_string(f)}")
                print(f"???: {read_uints(f, 4)}")
                num_sprites = read_uint(f)
                print(f"Number of sprites of this type: {num_sprites}")
                for _ in range(num_sprites):
                    print(f"Sprite position: {read_floats(f, 3)}")
                    print(f"Something else (radius?): {read_float(f)}")
            print(f"\nNumber of Pepsi types: 0")

        print(f"\nFinished at {hex(f.tell())}")
        next_part = f.read(4)
        print(f"Following part: {next_part}")
        assert next_part in (b"DYS\x01", b"STS\x00") or not next_part

        if mesh_type == "SHL":
            vtx_keys = ("position", "normal", "colour", "uv", "material", "deform_position", "deform_normal",
                        "octant_id1", "octant_id2", "octant_weight1", "octant_weight2")
        else:
            vtx_keys = ("position", "normal", "colour", "uv", "material")
        assert set(vtx_dict.keys()) == set(vtx_keys)

        fvtx_keys = vtx_keys + ("piece_id",)
        
        # Transform into flat, non-indexed data, 1:1 with face-vertices
        # TODO: this is kinda unintuitive and needs refactoring
        vtx_data = transposed(*(vtx_dict[key] for key in vtx_keys))
        fvtx_data = flatten_array(fvtx_dict["indices"], vtx_data)
        fvtx_attribs = transposed(*fvtx_data, container_type=list) + [fvtx_dict["piece_id"]]
        face_vertices = cls.Geometry(**dict(zip(fvtx_keys, fvtx_attribs)))

        textures_dir = os.path.dirname(f.name)
        if not os.path.isdir(textures_dir):
            textures_dir = None
    
        return Mesh(mesh_type, version, bounds, face_vertices, locators, hitboxes, textures_dir)
    
    def add_hitboxes_to_prim(self, prim):
        hitbox_root_path = prim.GetPath().AppendPath("hitboxes")
        hitbox_root_xform = UsdGeom.Xform.Define(prim.GetStage(), hitbox_root_path)
        hitbox_root_xform.MakeInvisible()
        for i, hitbox in enumerate(self._hitboxes):
            hitbox_prim_path = hitbox_root_path.AppendPath(f"hitbox_{i}")
            hitbox_cube = UsdGeom.Cube.Define(prim.GetStage(), hitbox_prim_path)
            UsdGeom.XformCommonAPI(hitbox_cube).SetTranslate(Gf.Vec3d(hitbox.centre))
            UsdGeom.XformCommonAPI(hitbox_cube).SetScale(hitbox.extent())
    
    @classmethod
    def get_hitboxes_from_geometry(cls, geometry):
        pieces = set(geometry.elements().values_of("piece_id").unique_elements())
        missing = set(range(16)) - pieces
        assert not missing, (f"SHL (armour) mesh does not include all the required pieces. "
                             f"Pieces {', '.join(str(x) for x in missing)} are missing. "
                             "Ensure that the \"piece_id\" attribute includes integers 0-15.")
        hitboxes = []
        for piece_id, face_vertices in geometry.elements().ordered_by("piece_id").split_by("piece_id"):
            hitboxes.append(cls.Bounds.from_points(face_vertices.values_of("position"),
                                                   padding=Gf.Vec3f(1., 0., 1.)))
        return hitboxes

    @staticmethod
    def sanitise_name(name):
        # USD prim names have some restrictions
        if not name or name[0].isnumeric():
            name = "_" + name
        name = name.replace(" ", "_")
        name = name.replace("-", "_")
        name = name.replace(".", "_")
        return name
    
    def export_to_usd(self, filepath, merge_vertices=False):
        print(f"Exporting to USD: {filepath}".center(80, "="))
        # Create a new USD stage
        stage = Usd.Stage.CreateNew(filepath)
        stage.SetMetadata("upAxis", "Y")

        # Create a single mesh node
        name = self.sanitise_name(os.path.splitext(os.path.basename(filepath))[0])
        mesh = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")

        # We use "fvtx" or "vtx" to denote whether the array items are one per face-vertex or one per vertex
        fvtx_positions = self.geometry.attribute("position")
        fvtx_normals = self.geometry.attribute("normal")
        fvtx_colours = self.geometry.attribute("colour")
        fvtx_uvs = self.geometry.attribute("uv")
        fvtx_materials = self.geometry.attribute("material")
        # TODO: apply merge_vertices to the new geometry iterators system
        fvtx_p_indices, points = unflatten_array(fvtx_positions, deduplicate=merge_vertices)
        # Now we have an index for each face-vertex which maps it to a *point* - NOT the same as RWM!
        assert len(fvtx_p_indices) % 3 == 0  # We should be able to construct triangles from these

        tri_materials = to_tri_array(fvtx_materials)
        #tri_m_indices, materials = unflatten_array(tri_materials)
        mat_to_tri_indices = inverse_index(tri_materials)

        # Create face sets for each material
        mat_names = set()
        for material, tri_indices in mat_to_tri_indices.items():
            mat_name, spec_map, mat_colour = material

            mat_name = self.sanitise_name(mat_name)

            # Make sure it's unique
            orig_mat_name = mat_name
            index = 0
            while mat_name in mat_names:
                index += 1
                mat_name = orig_mat_name + str(index)

            mat_names.add(mat_name)

            geom_subset = UsdGeom.Subset.CreateGeomSubset(mesh, mat_name, "face", tri_indices, "materialBind")
            UsdShade.MaterialBindingAPI.Apply(geom_subset.GetPrim())

            # Create a Material node under /Root/Materials/
            material_prim = UsdShade.Material.Define(stage, f"/_materials/{mat_name}")

            # Create a simple shader (you can extend this to support more complex material properties)
            shader_prim = UsdShade.Shader.Define(stage, f"/_materials/{mat_name}/Principled_BSDF")
            shader_prim.CreateIdAttr("UsdPreviewSurface")

            # Set some default PBR values (this can be extended)
            if self._textures_dir:
                tex_path = os.path.join(self._textures_dir, mat_name.lstrip("_") + ".tga")
                if not os.path.isfile(tex_path):
                    tex_path = os.path.join(self._textures_dir, mat_name.lstrip("_") + ".bmp")
                if not os.path.isfile(tex_path):
                    tex_path = None
            else:
                tex_path = None

            diffuse_colour_input = shader_prim.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            if tex_path:
                texture_prim_path = material_prim.GetPath().AppendPath("diffuseTexture")
                texture_shader = UsdShade.Shader.Define(stage, texture_prim_path)
                texture_shader.CreateIdAttr("UsdUVTexture")
                texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path.replace("\\", "/"))
                texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                texture_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                texture_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                uv_primvar_prim_path = material_prim.GetPath().AppendPath("uvReader")
                uv_reader_shader = UsdShade.Shader.Define(stage, uv_primvar_prim_path)
                uv_reader_shader.CreateIdAttr("UsdPrimvarReader_float2")
                uv_reader_shader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                uv_output = uv_reader_shader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_output)
                texture_output = texture_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Color3d)
                diffuse_colour_input.ConnectToSource(texture_output)
            else:
                diffuse_colour_input.Set(Gf.Vec3f(*mat_colour[:3]))  # OwO
            shader_prim.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader_prim.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

            # Bind shader to material
            material_prim.CreateSurfaceOutput().ConnectToSource(UsdShade.ConnectableAPI(shader_prim), "surface")

            # Bind material to the mesh faces that use it
            UsdShade.MaterialBindingAPI(geom_subset).Bind(material_prim)

        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())

        # Set vertex positions, normals, uvs, and face indices in the mesh
        mesh.GetPointsAttr().Set(points)
        mesh.GetNormalsAttr().Set(fvtx_normals)
        mesh.GetFaceVertexIndicesAttr().Set(fvtx_p_indices)
        mesh.GetFaceVertexCountsAttr().Set([3] * len(tri_materials))

        # Set UVs (texture coordinates) in a Primvar
        primvars = UsdGeom.PrimvarsAPI(mesh)
        st_primvar = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                            interpolation="faceVarying")
        st_primvar.Set(fvtx_uvs)

        piece_id_primvar = primvars.CreatePrimvar("piece_id", Sdf.ValueTypeNames.IntArray,
                                                  interpolation="faceVarying")
        piece_id_primvar.Set(list(self.geometry.elements().values_of("piece_id")))

        def convert_vtx_attrib_to_primvar(attrib_name, value_type):
            primvar = primvars.CreatePrimvar(attrib_name, value_type, interpolation="vertex")
            attrib_values = list(self.geometry.elements().values_of(attrib_name))#.unique_elements())
            assert len(points) == len(attrib_values)
            primvar.Set(attrib_values)
        
        for attrib_name, value_type in [
            ("deform_position", Sdf.ValueTypeNames.Point3fArray),
            ("deform_normal", Sdf.ValueTypeNames.Normal3fArray),
            ("octant_id1", Sdf.ValueTypeNames.IntArray),
            ("octant_id2", Sdf.ValueTypeNames.IntArray),
            ("octant_weight1", Sdf.ValueTypeNames.FloatArray),
            ("octant_weight2", Sdf.ValueTypeNames.FloatArray),
        ]:
            if self.geometry.has_attribute(attrib_name):
                convert_vtx_attrib_to_primvar(attrib_name, value_type)

        # TODO: do the same for colours

        if self._hitboxes:
            self.add_hitboxes_to_prim(mesh.GetPrim())

        for loc_name, matrix in self._locators.items():
            gf_matrix = Gf.Matrix4d(matrix)
            xform_prim = UsdGeom.Xform.Define(stage, f'/{loc_name}')
            xform = UsdGeom.Xform(xform_prim)
            xform_op = xform.AddTransformOp()
            xform_op.Set(gf_matrix)

        # Save the stage
        stage.GetRootLayer().Save()

        print(f"Mesh successfully exported to USD: {filepath}")
    
    @classmethod
    def get_material(cls, material_prim):
        mat_name = material_prim.GetName()
        spec_map = ""  # TODO
        mat_colour = (1.,) * 4  # TODO: proper colours
        return (mat_name, spec_map, mat_colour)
    
    @classmethod
    def get_geometry(cls, mesh_prim):
        mesh = UsdGeom.Mesh(mesh_prim)
        transform = mesh.ComputeLocalToWorldTransform(time=Usd.TimeCode.Default())
        if not all(x == 3 for x in mesh.GetFaceVertexCountsAttr().Get()):
            raise RuntimeError("This script does not support faces with >3 vertices - please "
                               "triangulate your mesh first.")
        
        usd_points = [transform.Transform(p) for p in mesh.GetPointsAttr().Get()]
        usd_fvtx_indices = mesh.GetFaceVertexIndicesAttr().Get()
        usd_normals = [transform.TransformDir(n) for n in mesh.GetNormalsAttr().Get()]
        positions = flatten_array(usd_fvtx_indices, usd_points)
        colours = [(1., 1., 1., 1.) for _ in usd_fvtx_indices]  # TODO

        attrib_dict = defaultdict(list)
        attrib_dict["position"] = positions
        attrib_dict["normal"] = usd_normals
        attrib_dict["colour"] = colours

        primvars = UsdGeom.PrimvarsAPI(mesh_prim)
        def facevertex_data_of(primvar_name):
            primvar = primvars.GetPrimvar(primvar_name)
            interp = primvar.GetInterpolation()
            if interp == "vertex":
                fvtx_data = flatten_array(usd_fvtx_indices, primvar.Get())
            elif interp == "uniform":
                fvtx_data = to_fvtx_array(primvar.Get())
            elif interp == "faceVarying":
                fvtx_data = primvar.Get()
            else:
                raise ValueError(f"Unrecognised primvar interpolation type: {interp}")
            
            if primvar.GetTypeName() == Sdf.ValueTypeNames.Point3fArray:
                fvtx_data = [transform.Transform(p) for p in fvtx_data]
            elif primvar.GetTypeName() == Sdf.ValueTypeNames.Normal3fArray:
                fvtx_data = [transform.TransformDir(n) for n in fvtx_data]

            return fvtx_data

        for attrib_name, primvar_name in {
            "uv": "st",
            "deform_position": "deform_position",
            "deform_normal": "deform_normal",
            "piece_id": "piece_id",
            "octant_id1": "octant_id1",
            "octant_id2": "octant_id2",
            "octant_weight1": "octant_weight1",
            "octant_weight2": "octant_weight2"
        }.items():
            if primvars.HasPrimvar(primvar_name):
                attrib_dict[attrib_name] = facevertex_data_of(primvar_name)

        if "piece_id" not in attrib_dict:
            attrib_dict["piece_id"] = [0] * len(usd_fvtx_indices)

        # Handle materials
        mat_api = UsdShade.MaterialBindingAPI(mesh_prim)
        default_material_prim = mat_api.GetDirectBinding().GetMaterial().GetPrim()
        if default_material_prim:
            default_material = cls.get_material(default_material_prim)
        else:
            default_material = None
        materials = [default_material] * len(usd_fvtx_indices)

        # Override default material from subsets
        geom_subsets = UsdGeom.Subset.GetGeomSubsets(mesh)
        for subset in geom_subsets:
            tri_indices = subset.GetIndicesAttr().Get()
            material = cls.get_material(subset.GetPrim())

            for tri_index in tri_indices:
                start = tri_index * 3
                materials[start] = material
                materials[start + 1] = material
                materials[start + 2] = material
        assert all(materials)
        attrib_dict["material"] = materials

        return Mesh.Geometry(**attrib_dict)

    @classmethod
    def import_from_usd(cls, filepath, mesh_type="MSH"):
        print(f"Importing from USD: {filepath}".center(80, "="))
        # Load the USD stage
        stage = Usd.Stage.Open(filepath)

        geometry = None
        locators = {}

        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                if not geometry:
                    geometry = cls.get_geometry(prim)
                else:
                    geometry.merge(cls.get_geometry(prim))
            
            if prim.IsA(UsdGeom.Xform) and not prim.GetChildren():
                loc_name = prim.GetName()
                loc = UsdGeom.Xform(prim)
                matrix = loc.ComputeLocalToWorldTransform(time=Usd.TimeCode.Default())
                locators[loc_name] = [vec for vec in matrix]
        
        positions = set(geometry.attribute("position"))
        bounds = Mesh.Bounds.from_points(positions)
        if mesh_type == "SHL":
            hitboxes = cls.get_hitboxes_from_geometry(geometry)
        else:
            hitboxes = []

        print(f"Mesh successfully imported from USD: {filepath}")
        return Mesh(mesh_type, Mesh.type_version_defaults[mesh_type],
                    bounds, geometry, locators, hitboxes, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="The mesh file to query or convert")
    parser.add_argument("out_path", nargs="?", help="The path of the converted output file")
    parser.add_argument("--mesh-type", default="MSH", choices=Mesh.supported_types, help="Mesh "
                        "type to assume when converting from USD.")
    parser.add_argument("--merge-vertices", required=False, action="store_true",
                        help="Deduplicate points in the output USD mesh. This uses less memory/"
                        "storage, but may mess up normals on \"double-sided\" geometry where two "
                        "faces use the same points.")
    args = parser.parse_args()

    basename, in_extension = os.path.splitext(args.in_path)
    if in_extension == ".rwm":
        with open(args.in_path, "rb") as in_file:
            m = Mesh.read_from_rwm(in_file)
    else:
        m = Mesh.import_from_usd(args.in_path, args.mesh_type)

    if args.out_path:
        out_basename, out_extension = os.path.splitext(args.out_path)
        if out_extension == ".rwm":
            with open(args.out_path, "wb") as out_file:
                m.write_to_rwm(out_file)
        else:
            m.export_to_usd(args.out_path, merge_vertices=args.merge_vertices)
        print(f"Successfully converted to {args.out_path}")