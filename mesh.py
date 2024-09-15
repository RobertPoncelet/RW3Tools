import argparse
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from pprint import pprint

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

# Read functions

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
def perpendicular_slices(*arrays, container_type=tuple):
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
    return [[item] * 3 for item in tri_array]


class Mesh:
    def __init__(self, face_vertices, locators):
        face_vertices.validate()
        self._face_vertices = face_vertices
        self._locators = locators

    @dataclass
    class FaceVertices:
        positions: list[tuple]
        normals: list[tuple]
        colours: list[tuple]
        uvs: list[tuple]
        materials: list[tuple]

        def validate(self):
            # Let's do some sanity checks
            # These *should* all have 1 item per face-vertex
            num = len(self.positions)
            assert num % 3 == 0
            assert num == len(self.normals) == len(self.colours) == len(self.uvs) == len(self.materials)
            to_tri_array(self.materials)
        
        def as_tuple(self):
            return self.positions, self.normals, self.colours, self.uvs, self.materials

    def write_to_rwm(self, f):
        print("Writing to RWM".center(80, "="))
        # We use "fvtx" or "vtx" to denote whether array items are per face-vertex or per vertex
        fvtx_data = perpendicular_slices(*self._face_vertices.as_tuple())
        # Sort by material, since they need to be contiguous
        fvtx_data = sorted(fvtx_data, key=lambda fvtx: fvtx[4])
        # Make an index for each face-vertex which maps it to a vertex
        fvtx_indices, vtx_data = unflatten_array(fvtx_data)
        assert len(fvtx_indices) % 3 == 0  # We should be able to construct triangles from these
        vtx_positions, vtx_normals, vtx_colours, vtx_uvs, vtx_materials = perpendicular_slices(*vtx_data, container_type=list)

        # We're relying on this producing the same material order as that of vtx_data
        ordered_materials = sorted(set(vtx_materials))
        # Now we need to obtain the numbers of face-vertices and vertices for each material
        mat_to_fvtx_indices = defaultdict(list)
        for fvtx_index in fvtx_indices:
            material = vtx_data[fvtx_index][4]
            mat_to_fvtx_indices[material].append(fvtx_index)
        mat_to_vertices = defaultdict(list)
        for vertex in vtx_data:
            mat_to_vertices[vertex[4]].append(vertex)

        f.write(b"MSH\x01")
        write_uint(f, len(self._locators))
        write_uint(f, len(ordered_materials))
        write_uint(f, len(vtx_data))
        write_uint(f, len(fvtx_indices) // 3)  # Number of triangles
        write_uint(f, 1)  # ???

        minx = min(v[0] for v in vtx_positions)
        miny = min(v[1] for v in vtx_positions)
        minz = min(v[2] for v in vtx_positions)
        maxx = max(v[0] for v in vtx_positions)
        maxy = max(v[1] for v in vtx_positions)
        maxz = max(v[2] for v in vtx_positions)
        write_floats(f, (minx, miny, minz))
        write_floats(f, (maxx, maxy, maxz))
        write_floats(f, ((minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2))
        write_float(f, 30.)  # Mass?

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
        
        for material in ordered_materials:
            mat_vertices = mat_to_vertices[material]
            write_uint(f, 0)
            write_uint(f, len(mat_vertices))
            for vertex in mat_vertices:
                position, normal, colour, uv, _ = vertex
                write_floats(f, position)
                write_floats(f, normal)
                assert len(colour) == 4
                write_uchars(f, tuple(int(c*255) for c in colour))
                write_floats(f, (uv[0], -uv[1]))  # Flip V

            mat_fvtx_indices = mat_to_fvtx_indices[material]
            write_uints(f, (1, 0))
            write_uint(f, len(mat_fvtx_indices) // 3)
            write_ushorts(f, mat_fvtx_indices)

    @classmethod
    def read_from_rwm(cls, f):
        print("Reading from RWM".center(80, "="))
        header = f.read(4)
        mesh_type = header[:3].decode("ascii")
        version = int(header[-1])
        assert mesh_type in ("MSH", "ARE")
        print(f"{mesh_type} mesh type, version {version}")
        num_locators = read_uint(f)
        print(f"Number of locators: {num_locators}")
        num_materials = read_uint(f)
        print(f"Number of materials: {num_materials}")
        print(f"Total number of verts: {read_uint(f)}")
        print(f"Total number of triangles: {read_uint(f)}")
        num_pieces = read_uint(f)
        print(f"Total number of mesh pieces: {num_pieces}")
        if mesh_type != "SHL":
            assert num_pieces == 1
        print(f"Bounding box min: {read_floats(f, 3)}")
        print(f"Bounding box max: {read_floats(f, 3)}")
        print(f"Another point (origin or centre of mass?): {read_floats(f, 3)}")
        print(f"Something else (mass?): {read_float(f)}")

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
                print(f"\nMaterial name: {mat_name}")
            else:
                print(f"\nNAMELESS MATERIAL at {hex(f.tell())}")
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

        fvtx_indices = []
        vtx_positions = []
        vtx_normals = []
        vtx_colours = []
        vtx_uvs = []
        vtx_materials = []

        for material in materials:
            assert read_uint(f) == 0
            num_verts = read_uint(f)
            print(f"Number of vertices for {material[0]}: {num_verts}")

            for _ in range(num_verts):
                vtx_positions.append(read_floats(f, 3))
                vtx_normals.append(read_floats(f, 3))
                vtx_colours.append(tuple(c / 255. for c in read_uchars(f, 4)))
                uv = read_floats(f, 2)
                vtx_uvs.append((uv[0], -uv[1]))  # Flip V
                vtx_materials.append(material)

            assert read_uints(f, 2) == (1, 0)
            num_tris = read_uint(f)
            print(f"Number of triangles: {num_tris}")
            fvtx_indices.extend(read_ushorts(f, num_tris*3))

        if mesh_type == "ARE":
            num_sprite_types = read_uint(f)
            print(f"\nNumber of sprite types: {num_sprite_types}")
            # Since I've only found one example of a sprite type, I'm not sure if they're ordered
            # as all sprite types followed by all sprites, or each array of sprites preceded by
            # their sprite type
            for _ in range(num_sprite_types):
                print(f"Sprite type name: {read_string(f)}")
                print(f"Some shit: {read_uints(f, 4)}")
                num_sprites = read_uint(f)
                print(f"Number of sprites of this type: {num_sprites}")
                for _ in range(num_sprites):
                    print(f"Sprite position: {read_floats(f, 3)}")
                    print(f"Something else (radius?): {read_float(f)}")

        print(f"\nFinished at {hex(f.tell())}")
        next_part = f.read(4)
        print(f"Following part: {next_part}")
        assert next_part in (b"DYS\x01", b"STS\x00") or not next_part
        
        # Transform into flat, non-indexed data, 1:1 with face-vertices
        vtx_data = perpendicular_slices(vtx_positions, vtx_normals, vtx_colours, vtx_uvs, vtx_materials)
        fvtx_data = flatten_array(fvtx_indices, vtx_data)
        positions, normals, colours, uvs, materials = perpendicular_slices(*fvtx_data, container_type=list)

        return Mesh(cls.FaceVertices(positions, normals, colours, uvs, materials), locators)
    
    def export_to_usd(self, filepath, merge_vertices=False):
        print(f"Exporting to USD: {filepath}".center(80, "="))
        # Create a new USD stage
        stage = Usd.Stage.CreateNew(filepath)
        stage.SetMetadata("upAxis", "Y")

        # Create a single mesh node
        name = os.path.splitext(os.path.basename(filepath))[0]
        mesh = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")

        # We use "fvtx" or "vtx" to denote whether the array items are one per face-vertex or one per vertex
        fvtx_positions, fvtx_normals, fvtx_colours, fvtx_uvs, fvtx_materials = self._face_vertices.as_tuple()
        fvtx_p_indices, points = unflatten_array(fvtx_positions, deduplicate=merge_vertices)
        # Now we have an index for each face-vertex which maps it to a *point* - NOT the same as RWM!
        assert len(fvtx_p_indices) % 3 == 0  # We should be able to construct triangles from these

        tri_materials = to_tri_array(fvtx_materials)
        #tri_m_indices, materials = unflatten_array(tri_materials)
        mat_to_tri_indices = inverse_index(tri_materials)

        # Create face sets for each material
        for material, tri_indices in mat_to_tri_indices.items():
            mat_name, spec_map, mat_colour = material
            geom_subset = UsdGeom.Subset.CreateGeomSubset(mesh, mat_name, "face", tri_indices)
            UsdShade.MaterialBindingAPI.Apply(geom_subset.GetPrim())

            # Create a Material node under /Root/Materials/
            material_prim = UsdShade.Material.Define(stage, f"/_materials/{mat_name}")

            # Create a simple shader (you can extend this to support more complex material properties)
            shader_prim = UsdShade.Shader.Define(stage, f"/_materials/{mat_name}/Principled_BSDF")
            shader_prim.CreateIdAttr("UsdPreviewSurface")

            # Set some default PBR values (this can be extended)
            shader_prim.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*mat_colour[:3]))  # OwO
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
        st_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",
                                                             Sdf.ValueTypeNames.TexCoord2fArray,
                                                             interpolation="faceVarying")
        st_primvar.Set(fvtx_uvs)

        # TODO: do the same for colours

        for loc_name, matrix in self._locators.items():
            gf_matrix = Gf.Matrix4d(matrix)
            xform_prim = UsdGeom.Xform.Define(stage, f'/{loc_name}')
            xform = UsdGeom.Xform(xform_prim)
            xform_op = xform.AddTransformOp()
            xform_op.Set(gf_matrix)

        # Save the stage
        stage.GetRootLayer().Save()

        print(f"Mesh successfully exported to USD: {filepath}")
    
    @staticmethod
    def get_texture_path(material_prim):
        # Get the UsdShadeMaterial representation of the prim
        material = UsdShade.Material(material_prim)

        # Find the surface output of the material
        surface_output = material.GetSurfaceOutput()

        if not surface_output:
            print(f"No surface output found for material {material_prim.GetPath()}")
            return None

        # Get the connected shader for the surface output
        connected_source = surface_output.GetConnectedSource()

        if not connected_source:
            print(f"No connected shader found for surface output of material {material_prim.GetPath()}")
            return None

        shader_prim = connected_source[0]  # The first element is the shader prim
        shader = UsdShade.Shader(shader_prim)

        # Iterate over the inputs of the shader to find any texture connections
        for input in shader.GetInputs():
            if input.GetTypeName() == "Asset":  # The texture file is stored as an Asset
                file_path = input.Get()
                if file_path:
                    print(f"Texture path: {file_path}")
                    return file_path
                else:
                    print(f"No texture path set for input {input.GetBaseName()}")
            elif input.HasConnectedSource():
                # Check if this input is connected to another shader (like a texture)
                connected_shader_source = input.GetConnectedSource()
                connected_shader_prim = connected_shader_source[0]
                connected_shader = UsdShade.Shader(connected_shader_prim)

                # Check if this shader is a texture shader (e.g., UsdUVTexture)
                shader_id = connected_shader.GetIdAttr().Get()
                if shader_id == "UsdUVTexture":
                    texture_input = connected_shader.GetInput("file")
                    if texture_input:
                        texture_path = texture_input.Get()
                        print(f"Texture path: {texture_path}")
                        return texture_path

        print(f"No texture found for material {material_prim.GetPath()}")
        return None
    
    @classmethod
    def get_material(cls, material_prim):
        mat_name = material_prim.GetName()
        spec_map = ""  # TODO
        mat_colour = (1.,) * 4  # TODO: proper colours
        return (mat_name, spec_map, mat_colour)
    
    @classmethod
    def get_face_vertices(cls, mesh_prim):
        mesh = UsdGeom.Mesh(mesh_prim)
        transform = mesh.ComputeLocalToWorldTransform(time=Usd.TimeCode.Default())
        
        usd_points = [transform.Transform(p) for p in mesh.GetPointsAttr().Get()]
        usd_fvtx_indices = mesh.GetFaceVertexIndicesAttr().Get()
        usd_normals = [transform.TransformDir(n) for n in mesh.GetNormalsAttr().Get()]
        usd_uvs = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st").Get()
        assert len(usd_fvtx_indices) == len(usd_normals) == len(usd_uvs)
        if not all(x == 3 for x in mesh.GetFaceVertexCountsAttr().Get()):
            raise RuntimeError("This script does not support faces with >3 vertices - please "
                               "triangulate your mesh first.")

        positions = flatten_array(usd_fvtx_indices, usd_points)
        colours = [(1., 1., 1., 1.) for _ in usd_fvtx_indices]  # TODO

        # Handle materials
        mat_api = UsdShade.MaterialBindingAPI(mesh_prim)
        default_material_prim = mat_api.GetDirectBinding().GetMaterial().GetPrim()
        default_material = cls.get_material(default_material_prim)
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

        positions = [tuple(p) for p in positions]
        normals = [tuple(n) for n in usd_normals]
        uvs = [tuple(uv) for uv in usd_uvs]

        return positions, normals, colours, uvs, materials

    @classmethod
    def import_from_usd(cls, filepath):
        print(f"Importing from USD: {filepath}".center(80, "="))
        # Load the USD stage
        stage = Usd.Stage.Open(filepath)

        face_vertices = [], [], [], [], []
        locators = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                for i, fvtx_attribute_list in enumerate(cls.get_face_vertices(prim)):
                    face_vertices[i].extend(fvtx_attribute_list)
            
            if prim.IsA(UsdGeom.Xform) and not prim.GetChildren():
                loc_name = prim.GetName()
                loc = UsdGeom.Xform(prim)
                matrix = loc.ComputeLocalToWorldTransform(time=Usd.TimeCode.Default())
                locators[loc_name] = [vec for vec in matrix]

        print(f"Mesh successfully imported from USD: {filepath}")
        return Mesh(cls.FaceVertices(*face_vertices), locators)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="The mesh file to query or convert")
    parser.add_argument("out_path", nargs="?", help="The path of the converted output file")
    parser.add_argument("--merge-vertices", required=False, action="store_true",
                        help="Deduplicate points in the output USD mesh. This uses less memory/"
                        "storage, but may mess up normals on \"double-sided\" geometry where two "
                        "faces use the same points.")
    args = parser.parse_args()

    print(f"Converting {args.in_path} -> {args.out_path}".center(80, "="))

    basename, in_extension = os.path.splitext(args.in_path)
    if in_extension == ".rwm":
        with open(args.in_path, "rb") as in_file:
            m = Mesh.read_from_rwm(in_file)
    else:
        m = Mesh.import_from_usd(args.in_path)

    if args.out_path:
        out_basename, out_extension = os.path.splitext(args.out_path)
        if out_extension == ".rwm":
            with open(args.out_path, "wb") as out_file:
                m.write_to_rwm(out_file)
        else:
            m.export_to_usd(args.out_path, merge_vertices=args.merge_vertices)