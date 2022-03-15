import open3d as o3d
import json
import numpy as np

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #     R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

    def merge_cylinder_segments(self):

         vertices_list = [np.asarray(mesh.vertices) for mesh in self.cylinder_segments]
         triangles_list = [np.asarray(mesh.triangles) for mesh in self.cylinder_segments]
         triangles_offset = np.cumsum([v.shape[0] for v in vertices_list])
         triangles_offset = np.insert(triangles_offset, 0, 0)[:-1]
        
         vertices = np.vstack(vertices_list)
         triangles = np.vstack([triangle + offset for triangle, offset in zip(triangles_list, triangles_offset)])
        
         merged_mesh = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(vertices), 
                                                 o3d.open3d.utility.Vector3iVector(triangles))
         color = self.colors if self.colors.ndim == 1 else self.colors[0]
         merged_mesh.paint_uniform_color(color)
         self.cylinder_segments = [merged_mesh]


def visualize_cameras(colored_camera_dicts, camera_size=1., 
                        geometry_file=None, geometry_type='mesh',
                        auto_adjust_camera_size=False):
    if auto_adjust_camera_size:
        assert geometry_file is not None and os.path.isfile(geometry_file), 'must specify a valid geometry path if auto_adjust_camera_size is true'

    things_to_draw = []

    coord_frame_radius = 0.5
    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
            points = np.asarray(geometry.vertices)
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
            points = np.asarray(geometry.points)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)
    
        # min_bound = geometry.get_min_bound()
        # max_bound = geometry.get_max_bound()

        min_x, max_x = np.percentile(points[:, 0], (1, 99))
        min_y, max_y = np.percentile(points[:, 1], (1, 99))
        min_z, max_z = np.percentile(points[:, 2], (2, 98))        

        min_bound = np.array([min_x, min_y, min_z])
        max_bound = np.array([max_x, max_y, max_z])
        diag = np.linalg.norm(max_bound - min_bound)
        # coord_frame_origin = (min_bound + max_bound) / 2.
        coord_frame_radius = diag / 2.
        camera_size = diag

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_radius, origin=[0., 0., 0.])
    things_to_draw.append(coord_frame)

    # for simplicity, we just visulize each satellite camera as a directional line originiating from (0, 0, 0)
    N = 0
    for color, camera_dict in colored_camera_dicts:
        N += len(camera_dict)
    merged_points = np.zeros((N*2, 3))   
    merged_lines = np.zeros((N, 2)).astype(int) 
    merged_colors = np.zeros((N, 3))
    idx = 0
    for color, camera_dict in colored_camera_dicts:
        for img_name in sorted(camera_dict.keys()):
            K = np.array(camera_dict[img_name]['K']).reshape((4, 4))
            W2C = np.array(camera_dict[img_name]['W2C']).reshape((4, 4))
            C2W = np.linalg.inv(W2C)
            img_size = camera_dict[img_name]['img_size']

            # merged_points[idx*2, :] = np.array([0., 0., 0.])
            cam_dir = C2W[:3, 3] / np.linalg.norm(C2W[:3, 3]) 
            merged_points[idx*2+1, :] = cam_dir * camera_size
            merged_lines[idx, 0] = idx * 2
            merged_lines[idx, 1] = idx * 2 + 1
            merged_colors[idx, :] = color
            
            idx += 1
    # lineset = o3d.geometry.LineSet()
    # lineset.points = o3d.utility.Vector3dVector(merged_points)
    # lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    # lineset.colors = o3d.utility.Vector3dVector(merged_colors)
    # things_to_draw.append(lineset)

    linemesh = LineMesh(merged_points, merged_lines, merged_colors, radius=4)
    linemesh.merge_cylinder_segments()
    things_to_draw.append(linemesh.cylinder_segments[0])

    o3d.visualization.draw_geometries(things_to_draw)


if __name__ == '__main__':
    import os

    base_dir = './example_visualize_cameras'

    train_cam_dict = json.load(open(os.path.join(base_dir, 'cam_dict.json')))
    # test_cam_dict = json.load(open(os.path.join(base_dir, 'test/cam_dict_norm.json')))
    # path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
    colored_camera_dicts = [([0, 1, 0], train_cam_dict),
                            # ([0, 0, 1], test_cam_dict),
                            # ([1, 1, 0], path_cam_dict)
                            ]

    geometry_file = os.path.join(base_dir, 'kai_points.ply')
    geometry_type = 'pointcloud'

    visualize_cameras(colored_camera_dicts, 
                      geometry_file=geometry_file, geometry_type=geometry_type, auto_adjust_camera_size=True)