import sys, os, struct
from functools import reduce
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf


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
    file.write(struct.pack("<" + (format * len(values)), values))


class Mesh:
    def __init__(self, submeshes):
        self._submeshes = submeshes

    def write_to_rwm(self, f):
        f.write(b"MSH\x01")
        write_uint(f, 0)  # num_mysteries
        write_uint(f, len(self._submeshes))
        all_positions = reduce(lambda a, b: a.extend(b), [submesh[2] for submesh in self._submeshes])
        all_indices = reduce(lambda a, b: a.extend(b), [submesh[6] for submesh in self._submeshes])
        write_uint(f, len(all_positions))
        write_uint(f, len(all_indices) / 3)
        write_uint(f, 1)  # ???
        minx = min(v[0] for v in all_positions)
        miny = min(v[1] for v in all_positions)
        minz = min(v[2] for v in all_positions)
        maxx = max(v[0] for v in all_positions)
        maxy = max(v[1] for v in all_positions)
        maxz = max(v[2] for v in all_positions)
        write_floats(f, (minx, miny, minz))
        write_floats(f, (maxx, maxy, maxz))
        write_floats(f, ((minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2))
        write_float(f, 30.)  # Mass?

        # TODO: write mysteries here

        for submesh in self._submeshes:
            material = submesh[0]
            spec_map = submesh[1]
            write_string(f, material)
            write_string(f, spec_map or "")
            f.write(b"\xcc\xcc\xcc\xff")
            write_float(f, 30.)  # ???
            write_uint(f, 0)  # Also dunno, flags?
        
        for submesh in self._submeshes:
            write_uint(f, 0)
            num_verts = len(submesh[2])  # Positions
            write_uint(f, num_verts)
            for i in range(num_verts):
                write_floats(f, submesh[2][i])
                write_floats(f, submesh[3][i])
                write_uchars(f, (int(c*255) for c in submesh[4][i]))
                uv = submesh[5][i]
                write_floats(f, (uv[0], -uv[1]))  # Flip V
            write_uints(f, (1, 0))
            write_uint(f, len(submesh[6])/3)
            write_ushorts(f, submesh[6])

    @classmethod
    def read_from_rwm_file(cls, f):
        assert f.read(4) == b"MSH\x01"
        num_mysteries = read_uint(f)
        print(f"Number of mysteries: {num_mysteries}")
        num_materials = read_uint(f)
        print(f"Number of materials: {num_materials}")
        print(f"Total number of verts: {read_uint(f)}")
        print(f"Total number of triangles: {read_uint(f)}")
        print(f"A low number, but NOT the number of materials: {read_uint(f)}")
        print(f"Bounding box min: {read_floats(f, 3)}")
        print(f"Bounding box max: {read_floats(f, 3)}")
        print(f"Another point (origin or centre of mass?): {read_floats(f, 3)}")
        print(f"Something else (mass?): {read_float(f)}")

        mysteries = []
        for _ in range(num_mysteries):
            mystery = read_string(f)
            print(f"\nMystery: {mystery}")
            mysteries.append(mystery)
            f.read(0x44)  # Who cares

        materials = []
        spec_maps = {}
        for _ in range(num_materials):
            material = read_string(f)
            print(f"\nMaterial: {material}")
            materials.append(material)
            spec_map = read_string(f)  # The specular map may be an empty string if not present
            if spec_map:
                print(f"Specular map: {spec_map}")
                spec_maps[material] = spec_map
            print(f"Bitfield maybe: {f.read(4)}")
            print(f"No idea: {read_float(f)}")
            print(f"Another number (flags?): {read_uint(f)}")

        submeshes = []
        num_indices = 0
        for material in materials:  # Each material is a string
            positions = []
            normals = []
            colours = []
            uvs = []
            assert read_uint(f) == 0
            num_verts = read_uint(f)
            print(f"Number of vertices for this material ({material}): {num_verts}")

            for _ in range(num_verts):
                positions.append(read_floats(f, 3))
                normals.append(read_floats(f, 3))
                colours.append((c / 255. for c in read_uchars(f, 4)))
                uv = read_floats(f, 2)
                uvs.append((uv[0], -uv[1]))  # Flip V

            assert read_uints(f, 2) == (1, 0)
            num_tris = read_uint(f)
            print(f"Number of triangles: {num_tris}")
            indices = [i - num_indices for i in read_ushorts(f, num_tris*3)]
            num_indices += max(indices) + 1
            submeshes.append((material, spec_maps.get(material), positions, normals, colours, uvs, indices))

        print(f"\nFinished at {hex(f.tell())}")
        next_part = f.read(4)
        print("Dynamics part", "follows" if next_part else "does NOT follow")
        assert next_part == b"DYS\x01" or not next_part

        return Mesh(submeshes)
    
    def export_to_usd(self, filepath):
        # Create a new USD stage
        stage = Usd.Stage.CreateNew(filepath)
        
        # Create the root Xform (transform) node for the mesh
        root_prim = UsdGeom.Xform.Define(stage, "/Root")

        # Create a single mesh node for all submeshes
        mesh_prim = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
        
        all_positions = []
        all_normals = []
        all_uvs = []
        all_indices = []
        face_vertex_counts = []
        face_sets = {}

        vertex_offset = 0

        # Iterate through submeshes to gather combined geometry data
        for submesh_idx, submesh in enumerate(self._submeshes):
            material, spec_map, positions, normals, colours, uvs, indices = submesh

            # Add vertices, normals, uvs to combined arrays
            all_positions.extend([Gf.Vec3f(p[0], p[1], p[2]) for p in positions])
            all_normals.extend([Gf.Vec3f(n[0], n[1], n[2]) for n in normals])
            all_uvs.extend([Gf.Vec2f(uv[0], uv[1]) for uv in uvs])

            # Adjust indices to account for the vertex offset
            adjusted_indices = [i + vertex_offset for i in indices]
            all_indices.extend(adjusted_indices)

            # Set face vertex counts (all triangles)
            face_vertex_counts.extend([3] * (len(indices) // 3))

            # Track faces for face sets (group faces by material)
            if material not in face_sets:
                face_sets[material] = []
            face_sets[material].extend(range(len(face_vertex_counts) - len(indices) // 3, len(face_vertex_counts)))

            # Update the vertex offset for the next submesh
            vertex_offset += len(positions)

        # Set vertex positions, normals, uvs, and face indices in the mesh
        mesh_prim.GetPointsAttr().Set(all_positions)
        mesh_prim.GetNormalsAttr().Set(all_normals)
        mesh_prim.GetFaceVertexIndicesAttr().Set(all_indices)
        mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)

        # Set UVs (texture coordinates) in a Primvar
        st_primvar = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar("st",
                                                                  Sdf.ValueTypeNames.TexCoord2fArray,
                                                                  interpolation="faceVarying")
        st_primvar.Set(all_uvs)

        # Create face sets for each material
        geom_subsets = {}
        for material, face_indices in face_sets.items():
            subset = UsdGeom.Subset.CreateGeomSubset(mesh_prim, material, "someElementType", face_indices)
            geom_subsets[material] = subset

        # Export materials and bind to the mesh
        self.export_usd_materials(stage, geom_subsets)

        # Save the stage
        stage.GetRootLayer().Save()

        print(f"Mesh successfully exported to USD: {filepath}")
    
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

    def export_usd_materials(self, stage, face_sets):
        for material_name, geom_subset in face_sets.items():
            # Create a Material node under /Root/Materials/
            material_prim = UsdShade.Material.Define(stage, f"/Root/Materials/{material_name}")

            # Create a simple shader (you can extend this to support more complex material properties)
            shader_prim = UsdShade.Shader.Define(stage, f"/Root/Materials/{material_name}/PBRShader")
            shader_prim.CreateIdAttr("UsdPreviewSurface")

            # Set some default PBR values (this can be extended)
            shader_prim.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
            shader_prim.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader_prim.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

            # Bind shader to material
            material_prim.CreateSurfaceOutput().ConnectToSource(UsdShade.ConnectableAPI(shader_prim), "surface")

            # Bind material to the mesh faces that use it
            #mesh_prim = stage.GetPrimAtPath("/Root/Mesh")
            UsdShade.MaterialBindingAPI(geom_subset).Bind(material_prim)

            print(material_prim, shader_prim)

    def import_from_usd(self, filepath):
        # Load the USD stage
        stage = Usd.Stage.Open(filepath)
        mesh_prim = stage.GetPrimAtPath("/Root/Mesh")
        mesh = UsdGeom.Mesh(mesh_prim)
        
        positions = mesh.GetPointsAttr().Get()
        normals = mesh.GetNormalsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        uvs = mesh.GetPrimvar("st").Get()
        
        submeshes = []

        # Handle materials and face sets
        face_sets = mesh.GetFaceSets()
        for face_set_name in face_sets:
            material = face_set_name  # Use the face set name as the material
            face_set_indices = face_sets[face_set_name].Get()
            
            # Extract the submesh using indices from the face set
            submesh_indices = [indices[i] for i in face_set_indices]
            submesh_positions = [positions[i] for i in submesh_indices]
            submesh_normals = [normals[i] for i in submesh_indices]
            submesh_uvs = [uvs[i] for i in submesh_indices]
            
            submeshes.append((material, submesh_positions, submesh_normals, [], submesh_uvs, submesh_indices))
            print(submeshes[-1])

        print(f"Mesh successfully imported from USD: {filepath}")
        return Mesh(submeshes)


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