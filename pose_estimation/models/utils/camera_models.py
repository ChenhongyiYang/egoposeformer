# Author: Chenhongyi Yang

import torch


def unrealego_proj(local_3d, local_origin):
    num_views = 2
    polynomial_w2c = (
        541.084422, 133.996745, -53.833198, 60.96083, -24.78051, 12.451492,
        -30.240511, 26.90122, 116.38499, -133.991117, -141.904687, 184.05592,
        107.45616, -125.552875, -55.66342, 44.209519, 18.234651, -6.410899, -2.737066
    )
    image_center = (511.1183388444314, 510.8730105600536)
    raw_image_size = (1024, 1024)

    with torch.no_grad():
        cam_3d = local_3d[:, None].repeat(1, num_views, 1, 1)  # [B, V, J, 3]
        cam_3d = cam_3d + local_origin

        x = cam_3d[..., 0]  # [B, V, J]
        y = cam_3d[..., 1]  # [B, V, J]
        z = cam_3d[..., 2]  # [B, V, J]

        norm = torch.sqrt(x * x + y * y)
        theta = torch.atan(-z / norm)

        rho = sum(a * theta ** i for i, a in enumerate(polynomial_w2c))

        u = x / norm * rho + image_center[0]
        v = y / norm * rho + image_center[1]

        u = u / raw_image_size[1]
        v = v / raw_image_size[0]

        image_coor_2d = torch.stack((u, v), dim=-1)  # [B, V, J, 2]
        in_fov = (
                (image_coor_2d[..., 0] > 0)
                & (image_coor_2d[..., 1] > 0)
                & (image_coor_2d[..., 0] < 1)
                & (image_coor_2d[..., 1] < 1)
        )  # [B, V, J]
        image_coor_2d = image_coor_2d.clamp(min=0.0, max=1.0)
        return image_coor_2d, in_fov


projection_funcs = {
    'unrealego': unrealego_proj
}