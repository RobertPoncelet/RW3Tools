import sys, os, struct
from functools import reduce
from dataclasses import dataclass
from collections import defaultdict
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
from pprint import pprint


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

def unflatten_array(flat_array):
    indexed_array = []
    indices = []
    for item in flat_array:
        if item in indexed_array:
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


class Mesh:
    def __init__(self, positions, normals, colours, uvs, materials):
        self.positions = positions
        self.normals = normals
        self.colours = colours
        self.uvs = uvs
        self.materials = materials

    @dataclass
    class Triangle:
        points: tuple
        normals: tuple
        colours: tuple
        uvs: tuple

        # First 4 buffers will include all the vertices in the mesh so far, while the index buffer
        # will be just for this tri set
        @classmethod
        def from_buffers(cls, point_buffer, normal_buffer, colour_buffer, uv_buffer, index_buffer,
                         offset):
            i = index_buffer[offset]
            j = index_buffer[offset+1]
            k = index_buffer[offset+2]
            def item(buffer):
                return (buffer[i], buffer[j], buffer[k])
            return cls(item(point_buffer), item(normal_buffer), item(colour_buffer), item(uv_buffer))

    @dataclass
    class TriSet:
        material: str
        spec_map: str | None
        colour: tuple[float]
        triangles: list

        def num_verts(self):
            return len(self.triangles) * 3
        
        def num_tris(self):
            return len(self.triangles)
        
        def iterate_verts(self):
            for tri in self.triangles:
                for i in range(3):
                    yield tri.points[i], tri.normals[i], tri.colours[i], tri.uvs[i]
        
        def indices(self):
            # TODO: this will need to be changed once we implement vertex sharing
            return list(range(self.num_verts()))

        @classmethod
        def from_buffers(cls, material, spec_map, colour, point_buffer, normal_buffer,
                         colour_buffer, uv_buffer, index_buffer, offset):
            tris = []
            for i in range(0, len(index_buffer), 3):
                tris.append(Mesh.Triangle.from_buffers(point_buffer, normal_buffer, colour_buffer,
                                                      uv_buffer, index_buffer, i+offset))
            return cls(material, spec_map, colour, tris)

    def write_to_rwm(self, f):
        print("Writing to RWM".center(80, "="))
        # Unflatten, grouped by material
        # We use "fvtx" or "vtx" to denote whether the array items are one per face-vertex or one per vertex
        fvtx_data = perpendicular_slices(self.positions, self.normals, self.colours, self.uvs, self.materials)
        fvtx_indices, vtx_data = unflatten_array(fvtx_data)  # Now we have an index for each face-vertex which maps it to a vertex
        assert len(fvtx_indices) % 3 == 0  # We should be able to construct triangles from these
        vtx_positions, vtx_normals, vtx_colours, vtx_uvs, vtx_materials = perpendicular_slices(*vtx_data, container_type=list)
        
        # DEBUG
        pprint(fvtx_indices)
        pprint((vtx_positions, vtx_normals, vtx_colours, vtx_uvs, vtx_materials))

        material_to_fvtxindices = defaultdict(list)
        material_to_vertices = defaultdict(list)
        for fvtx_index in fvtx_indices:
            material = vtx_materials[fvtx_index]
            vertex = vtx_data[fvtx_index]
            material_to_fvtxindices[material].append(fvtx_index)
            if vertex not in material_to_vertices[material]:
                material_to_vertices[material].append(vertex)

        f.write(b"MSH\x01")
        write_uint(f, 0)  # Number of locators
        write_uint(f, len(material_to_fvtxindices))
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

        # TODO: write locators here

        for material in material_to_fvtxindices.keys():
            mat_name, spec_map, mat_colour = material
            assert type(spec_map) is str
            write_string(f, mat_name)
            write_string(f, spec_map)
            assert(len(mat_colour) == 4)
            write_uchars(f, tuple(int(c*255) for c in mat_colour))
            write_float(f, 30.)  # ???
            write_uint(f, 0)  # Also dunno, flags?
        
        for material, mat_fvtx_indices in material_to_fvtxindices.items():
            write_uint(f, 0)
            vertices = material_to_vertices[material]
            write_uint(f, len(vertices))
            for vertex in vertices:
                position, normal, colour, uv, _ = vertex
                write_floats(f, position)
                write_floats(f, normal)
                assert len(colour) == 4
                write_uchars(f, tuple(int(c*255) for c in colour))
                write_floats(f, (uv[0], -uv[1]))  # Flip V
            write_uints(f, (1, 0))
            write_uint(f, len(mat_fvtx_indices) // 3)
            write_ushorts(f, mat_fvtx_indices)

    @classmethod
    def read_from_rwm(cls, f):
        print("Reading from RWM".center(80, "="))
        assert f.read(4) == b"MSH\x01"
        num_locators = read_uint(f)
        print(f"Number of locators: {num_locators}")
        num_materials = read_uint(f)
        print(f"Number of materials: {num_materials}")
        print(f"Total number of verts: {read_uint(f)}")
        print(f"Total number of triangles: {read_uint(f)}")
        print(f"A low number, but NOT the number of materials: {read_uint(f)}")
        print(f"Bounding box min: {read_floats(f, 3)}")
        print(f"Bounding box max: {read_floats(f, 3)}")
        print(f"Another point (origin or centre of mass?): {read_floats(f, 3)}")
        print(f"Something else (mass?): {read_float(f)}")

        locators = []
        for _ in range(num_locators):
            locator = read_string(f)
            print(f"\nlocator: {locator}")
            locators.append(locator)
            f.read(0x44)  # Who cares

        materials = []
        for _ in range(num_materials):
            mat_name = read_string(f)
            print(f"\nMaterial name: {mat_name}")
            spec_map = read_string(f)  # The specular map may be an empty string if not present
            print(f"Specular map: {spec_map or '<no spec map>'}")
            mat_colour = tuple(c / 255. for c in read_uchars(f, 4))
            print(f"Material colour(?): {mat_colour}")
            print(f"No idea: {read_float(f)}")
            print(f"Another number (flags?): {read_uint(f)}")
            materials.append((mat_name, spec_map, mat_colour))

        fvtx_indices = []
        vtx_positions = []
        vtx_normals = []
        vtx_colours = []
        vtx_uvs = []
        vtx_materials = []

        for material in materials:  # Each material is a string
            assert read_uint(f) == 0
            num_verts = read_uint(f)
            print(f"Number of vertices for this material ({material}): {num_verts}")

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

        print(f"\nFinished at {hex(f.tell())}")
        next_part = f.read(4)
        print("Dynamics part", "follows" if next_part else "does NOT follow")
        assert next_part == b"DYS\x01" or not next_part

        # DEBUG
        pprint(fvtx_indices)
        pprint((vtx_positions, vtx_normals, vtx_colours, vtx_uvs, vtx_materials))
        
        # Transform into flat, non-indexed data, 1:1 with face-vertices
        vtx_data = perpendicular_slices(vtx_positions, vtx_normals, vtx_colours, vtx_uvs, vtx_materials)
        fvtx_data = flatten_array(fvtx_indices, vtx_data)
        positions, normals, colours, uvs, materials = perpendicular_slices(*fvtx_data, container_type=list)

        return Mesh(positions, normals, colours, uvs, materials)
    
    def export_to_usd(self, filepath):
        print(f"Exporting to USD: {filepath}".center(80, "="))
        # Create a new USD stage
        stage = Usd.Stage.CreateNew(filepath)
        
        # Create the root Xform (transform) node for the mesh
        root_prim = UsdGeom.Xform.Define(stage, "/Root")

        # Create a single mesh node for all submeshes
        mesh_prim = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

        positions = []
        normals = []
        colours = []
        uvs = []

        # Create face sets for each material
        start_face_index = 0
        for tri_set in self.tri_sets:
            face_indices = range(start_face_index, start_face_index + tri_set.num_tris())
            start_face_index += tri_set.num_tris()
            geom_subset = UsdGeom.Subset.CreateGeomSubset(mesh_prim, tri_set.material, "face", face_indices)

            mat_normals = []

            for pos, normal, colour, uv in tri_set.iterate_verts():
                positions.append(pos)
                normals.append(normal)
                mat_normals.append(normal)
                colours.append(colour)
                uvs.append(uv)

            print(f"export_to_usd {tri_set.material} normals: {mat_normals}")

            # Create a Material node under /Root/Materials/
            material_prim = UsdShade.Material.Define(stage, f"/Root/Materials/{tri_set.material}")

            # Create a simple shader (you can extend this to support more complex material properties)
            shader_prim = UsdShade.Shader.Define(stage, f"/Root/Materials/{tri_set.material}/Principled_BSDF")
            shader_prim.CreateIdAttr("UsdPreviewSurface")

            # Set some default PBR values (this can be extended)
            shader_prim.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tri_set.colour[:3]))  # OwO
            shader_prim.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader_prim.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

            # Bind shader to material
            material_prim.CreateSurfaceOutput().ConnectToSource(UsdShade.ConnectableAPI(shader_prim), "surface")

            # Bind material to the mesh faces that use it
            UsdShade.MaterialBindingAPI(geom_subset).Bind(material_prim)

        # Set vertex positions, normals, uvs, and face indices in the mesh
        mesh_prim.GetPointsAttr().Set(positions)
        mesh_prim.GetNormalsAttr().Set(normals)
        total_num_verts = sum(ts.num_verts() for ts in self.tri_sets)
        # TODO: this will need to be changed once we implement vertex sharing
        all_indices = list(range(total_num_verts)) #reduce(lambda a, b: a + b, [ts.indices() for ts in self.tri_sets])
        mesh_prim.GetFaceVertexIndicesAttr().Set(all_indices)
        total_num_tris = sum(ts.num_tris() for ts in self.tri_sets)
        mesh_prim.GetFaceVertexCountsAttr().Set([3] * total_num_tris)

        # Set UVs (texture coordinates) in a Primvar
        st_primvar = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar("st",
                                                                  Sdf.ValueTypeNames.TexCoord2fArray,
                                                                  interpolation="faceVarying")
        st_primvar.Set(uvs)

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
    def import_from_usd(cls, filepath):
        print(f"Importing from USD: {filepath}".center(80, "="))
        # Load the USD stage
        stage = Usd.Stage.Open(filepath)
        default_prim = stage.GetDefaultPrim()
        if default_prim:
            # Check if it's the Mesh itself or the Mesh's parent
            if default_prim.IsA(UsdGeom.Mesh):
                mesh_prim = default_prim
            else:
                mesh_prim = next(p for p in default_prim.GetChildren() if p.IsA(UsdGeom.Mesh))
        else:
            # Search for any Mesh
            mesh_prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
        mesh = UsdGeom.Mesh(mesh_prim)
        
        usd_points = mesh.GetPointsAttr().Get()
        usd_fvtx_indices = mesh.GetFaceVertexIndicesAttr().Get()
        usd_normals = mesh.GetNormalsAttr().Get()
        usd_uvs = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st").Get()
        assert len(usd_fvtx_indices) == len(usd_normals) == len(usd_uvs)
        assert all(x == 3 for x in mesh.GetFaceVertexCountsAttr().Get())
        
        tri_sets = []

        # Handle materials and tri sets
        geom_subsets = UsdGeom.Subset.GetGeomSubsets(mesh)
        for i, subset in enumerate(geom_subsets):
            material = subset.GetPrim().GetName()  # Use the subset name as the material

            face_indices = subset.GetIndicesAttr().Get()
            
            # Extract the tri set using indices from the geom subset
            triangles = []
            mat_normals = []
            for face_index in face_indices:
                start = face_index * 3
                positions = []
                normals = []
                colours = []
                uvs = []
                for fvtx_index in start, start+1, start+2:
                    pos_index = usd_fvtx_indices[fvtx_index]
                    positions.append(usd_points[pos_index])
                    normals.append(usd_normals[fvtx_index])
                    mat_normals.append(usd_normals[fvtx_index])
                    colours.append((1., 1., 1., 1.))  # TODO
                    uvs.append(usd_uvs[fvtx_index])

                triangles.append(cls.Triangle(positions, normals, colours, uvs))
            print(f"import_from_usd {material} normals: {mat_normals}")
            # TODO: proper colours
            r = [1., 0., 0.][i]
            g = [0., 1., 0.][i]
            b = [0., 0., 1.][i]
            tri_sets.append(cls.TriSet(material, None, (r, g, b, 1.), triangles))

        print(f"Mesh successfully imported from USD: {filepath}")
        return Mesh(tri_sets)


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    print(f"Converting {in_path} => {out_path}".center(80, "="))
    basename, in_extension = os.path.splitext(in_path)
    if in_extension == ".rwm":
        with open(in_path, "rb") as in_file:
            m = Mesh.read_from_rwm(in_file)
            m.export_to_usd(out_path)
    else:
        m = Mesh.import_from_usd(in_path)
        with open(out_path, "wb") as out_file:
            m.write_to_rwm(out_file)