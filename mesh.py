import sys, os, struct
from functools import reduce
from dataclasses import dataclass
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
from pprint import pprint


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


class Mesh:
    def __init__(self, tri_sets: list):
        self.tri_sets = tri_sets

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
        
    def positions(self):
        for tri_set in self.tri_sets:
            for pos, _, _, _ in tri_set.iterate_verts():
                yield pos

    def write_to_rwm(self, f):
        f.write(b"MSH\x01")
        write_uint(f, 0)  # num_locators
        write_uint(f, len(self.tri_sets))
        write_uint(f, sum(fs.num_verts() for fs in self.tri_sets))
        write_uint(f, sum(fs.num_tris() for fs in self.tri_sets))
        write_uint(f, 1)  # ???
        minx = min(v[0] for v in self.positions())
        miny = min(v[1] for v in self.positions())
        minz = min(v[2] for v in self.positions())
        maxx = max(v[0] for v in self.positions())
        maxy = max(v[1] for v in self.positions())
        maxz = max(v[2] for v in self.positions())
        write_floats(f, (minx, miny, minz))
        write_floats(f, (maxx, maxy, maxz))
        write_floats(f, ((minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2))
        write_float(f, 30.)  # Mass?

        # TODO: write locators here

        for tri_set in self.tri_sets:
            write_string(f, tri_set.material)
            write_string(f, tri_set.spec_map or "")
            assert(len(tri_set.colour) == 4)
            write_uchars(f, tuple(int(c*255) for c in tri_set.colour))
            write_float(f, 30.)  # ???
            write_uint(f, 0)  # Also dunno, flags?
        
        for tri_set in self.tri_sets:
            write_uint(f, 0)
            write_uint(f, tri_set.num_verts())
            mat_normals = []
            for pos, normal, colour, uv in tri_set.iterate_verts():
                write_floats(f, pos)
                write_floats(f, normal)
                write_uchars(f, tuple(int(c*255) for c in colour))
                write_floats(f, (uv[0], -uv[1]))  # Flip V
                mat_normals.append(normal)
            print(f"write_to_rwm {tri_set.material} normals: {mat_normals}")
            write_uints(f, (1, 0))
            write_uint(f, tri_set.num_tris())
            write_ushorts(f, tri_set.indices())

    @classmethod
    def read_from_rwm_file(cls, f):
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

        material_names = []
        spec_maps = {}
        mat_colours = {}
        for _ in range(num_materials):
            material = read_string(f)
            print(f"\nMaterial: {material}")
            material_names.append(material)
            spec_map = read_string(f)  # The specular map may be an empty string if not present
            if spec_map:
                print(f"Specular map: {spec_map}")
                spec_maps[material] = spec_map
            mat_colour = tuple(c / 255. for c in read_uchars(f, 4))
            mat_colours[material] = mat_colour
            print(f"Material colour(?): {mat_colour}")
            print(f"No idea: {read_float(f)}")
            print(f"Another number (flags?): {read_uint(f)}")

        tri_sets = []

        positions = []
        normals = []
        colours = []
        uvs = []

        for material in material_names:  # Each material is a string
            assert read_uint(f) == 0
            num_verts = read_uint(f)
            print(f"Number of vertices for this material ({material}): {num_verts}")

            mat_normals = []
            offset = len(positions)

            for _ in range(num_verts):
                positions.append(read_floats(f, 3))
                normals.append(read_floats(f, 3))
                mat_normals.append(normals[-1])
                colours.append((c / 255. for c in read_uchars(f, 4)))
                uv = read_floats(f, 2)
                uvs.append((uv[0], -uv[1]))  # Flip V

            print(f"read_from_rwm {material} normals: {mat_normals}")

            assert read_uints(f, 2) == (1, 0)
            num_tris = read_uint(f)
            print(f"Number of triangles: {num_tris}")
            indices = read_ushorts(f, num_tris*3)

            tri_set = cls.TriSet.from_buffers(material, spec_maps.get(material),
                                              mat_colours[material], positions, normals, colours,
                                              uvs, indices, offset)
            assert num_tris == tri_set.num_tris()
            tri_sets.append(tri_set)

        print(f"\nFinished at {hex(f.tell())}")
        next_part = f.read(4)
        print("Dynamics part", "follows" if next_part else "does NOT follow")
        assert next_part == b"DYS\x01" or not next_part

        return Mesh(tri_sets)
    
    def export_to_usd(self, filepath):
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
            m = Mesh.read_from_rwm_file(in_file)
            m.export_to_usd(out_path)
    else:
        m = Mesh.import_from_usd(in_path)
        with open(out_path, "wb") as out_file:
            m.write_to_rwm(out_file)