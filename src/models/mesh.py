import kaolin as kal
import torch

import copy

class Mesh:
    def __init__(self,obj_path, device):
        # from https://github.com/threedle/text2mesh

        if ".obj" in obj_path:
            try:
                mesh = kal.io.obj.import_mesh(obj_path, with_normals=True, with_materials=True)
            except:
                mesh = kal.io.obj.import_mesh(obj_path, with_normals=True, with_materials=False)

        elif ".off" in obj_path:
            mesh = kal.io.off.import_mesh(obj_path)
        else:
            raise ValueError(f"{obj_path} extension not implemented in mesh reader.")

        self.vertices = mesh.vertices.to(device)
        self.faces = mesh.faces.to(device)
        self.normals, self.face_area = self.calculate_face_normals(self.vertices, self.faces)
        print("debug Mesh.__init__")
        print("self.vertices.shape:", self.vertices.shape)
        print("self.faces.shape:", self.faces.shape)
        print("self.normals.shape:", self.normals.shape)
        print("self.face_area.shape:", self.face_area.shape)
        self.ft = mesh.face_uvs_idx  # indices into UVmap for every vertex of every face of shape (num_faces, face_size), each face has three values correponding to the indices of face vertices in the uv map. (face table)
        if self.ft is not None:
            print("self.ft.shape:", self.ft.shape)
        self.vt = mesh.uvs  # UV map coordinates of shape (num_uvs, 2), num_uvs should equal to num_vertice of the mesh. UV map maps each mesh vertex to the 2-dim uv coordinates. (vertex table)
        if self.vt is not None:
            print("self.vt.shape:", self.vt.shape)

    @staticmethod
    def calculate_face_normals(vertices: torch.Tensor, faces: torch.Tensor):
        """
        calculate per face normals from vertices and faces
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        e0 = v1 - v0
        e1 = v2 - v0
        n = torch.cross(e0, e1, dim=-1)
        twice_area = torch.norm(n, dim=-1)
        n = n / twice_area[:, None]
        return n, twice_area / 2

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices
        center = verts.mean(dim=0)
        verts -= center
        scale = torch.std(torch.norm(verts, p=2, dim=1))
        verts /= scale
        mesh.vertices = verts
        return mesh

    def normalize_mesh(self,inplace=False, target_scale=1, dy=0):
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices
        center = verts.mean(dim=0)
        verts = verts - center
        scale = torch.max(torch.norm(verts, p=2, dim=1))
        verts = verts / scale
        verts *= target_scale
        verts[:, 1] += dy
        mesh.vertices = verts
        return mesh

