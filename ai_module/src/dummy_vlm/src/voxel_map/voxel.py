from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.voxel_grid import voxel_grid
from torch_geometric.utils import scatter


class VoxelizedPointcloud:
    _INTERNAL_TENSORS = [
        "_points",
        "_features",
        "_weights",
        "_rgb",
        "_scale",
        "dim_mins",
        "dim_maxs",
        "_mins",
        "_maxs",
    ]

    _INIT_ARGS = ["voxel_size", "dim_mins", "dim_maxs", "feature_pool_method"]

    def __init__(
        self,
        voxel_size: float = 0.05,
        dim_mins: Optional[Tensor] = None,
        dim_maxs: Optional[Tensor] = None,
        feature_pool_method: str = "mean",
    ):
        """
        Args:
            voxel_size (float): Voxel size in each dimension.
            dim_mins (Optional[Tensor]): Minimum coordinates possible in voxel grid.
            dim_maxs (Optional[Tensor]): Maximum coordinates possible in voxel grid.
            feature_pool_method (str, optional): How to pool features within a voxel. One of 'mean', 'max', 'sum'. Defaults to 'mean'.
        """
        assert (dim_mins is None) == (dim_maxs is None)
        self.dim_mins = dim_mins
        self.dim_maxs = dim_maxs
        self.voxel_size = voxel_size
        self.feature_pool_method = feature_pool_method
        assert self.feature_pool_method in ["mean", "max", "sum"], f"Unknown feature pool method {feature_pool_method}"

        self.reset()

    def reset(self):
        """Resets internal tensors."""
        self._points, self._features, self._weights, self._rgb = None, None, None, None
        self._scale = None
        self._mins = self.dim_mins
        self._maxs = self.dim_maxs

    def add(
        self,
        points: Tensor,
        features: Optional[Tensor],
        rgb: Optional[Tensor],
        weights: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ):
        """Add a feature pointcloud to the voxel grid.

        Args:
            points (Tensor): N x 3 points to add to the voxel grid.
            features (Optional[Tensor]): N x D features associated with each point.
            rgb (Optional[Tensor]): N x 3 colors associated with each point.
            weights (Optional[Tensor], optional): Weights for each point. Defaults to None.
            scale (Optional[Tensor], optional): Scale for each point (x, y, z). Defaults to None.
        """
        if weights is None:
            weights = torch.ones_like(points[..., 0])
        device = features.device
        if points is not None:
            points = points.to(device)
        if weights is not None:
            weights = weights.to(device)
        if scale is not None:
            scale = scale.to(device)

        # Update voxel grid bounds
        pos_mins, _ = points.min(dim=0)
        pos_maxs, _ = points.max(dim=0)
        if self.dim_mins is not None:
            assert torch.all(self.dim_mins <= pos_mins), "Got points outside of user-defined 3D bounds"
        if self.dim_maxs is not None:
            assert torch.all(pos_maxs <= self.dim_maxs), "Got points outside of user-defined 3D bounds"

        if self._mins is None:
            self._mins, self._maxs = pos_mins, pos_maxs
        else:
            self._mins = torch.min(self._mins, pos_mins)
            self._maxs = torch.max(self._maxs, pos_maxs)

        if self._points is None:
            all_points, all_features, all_weights, all_rgb, all_scale = points, features, weights, rgb, scale
        else:
            all_points = torch.cat([self._points, points], dim=0)
            all_weights = torch.cat([self._weights, weights], dim=0)
            all_features = torch.cat([self._features, features], dim=0) if features is not None else None
            all_rgb = torch.cat([self._rgb, rgb], dim=0) if rgb is not None else None
            all_scale = torch.cat([self._scale, scale], dim=0) if scale is not None else None

        cluster_voxel_idx, cluster_consecutive_idx, _ = voxelize(
            all_points, voxel_size=self.voxel_size, start=self._mins, end=self._maxs
        )

        self._points, self._features, self._weights, self._rgb, self._scale = reduce_pointcloud(
            cluster_consecutive_idx,
            pos=all_points,
            features=all_features,
            weights=all_weights,
            rgbs=all_rgb,
            scales=all_scale,
            feature_reduce=self.feature_pool_method,
        )

    def get_idxs(self, points: Tensor) -> Tensor:
        """Returns voxel index (long tensor) for each point in points.

        Args:
            points (Tensor): N x 3

        Returns:
            cluster_voxel_idx (Tensor): The voxel grid index (long tensor) for each point in points.
            cluster_consecutive_idx (Tensor): Voxel grid reindexed to be consecutive (packed).
        """
        cluster_voxel_idx, cluster_consecutive_idx, _ = voxelize(points, self.voxel_size, start=self._mins, end=self._maxs)
        return cluster_voxel_idx, cluster_consecutive_idx

    def get_voxel_idx(self, points: Tensor) -> Tensor:
        """Returns voxel index (long tensor) for each point in points.

        Args:
            points (Tensor): N x 3

        Returns:
            Tensor: voxel index (long tensor) for each point in points
        """
        cluster_voxel_idx, _ = self.get_idxs(points)
        return cluster_voxel_idx

    def get_consecutive_cluster_idx(self, points: Tensor) -> Tensor:
        """Returns consecutive cluster index for each point in points.

        Args:
            points (Tensor): N x 3

        Returns:
            Tensor: consecutive cluster index for each point in points
        """
        _, cluster_consecutive_idx = self.get_idxs(points)
        return cluster_consecutive_idx

    def get_pointcloud(self) -> Tuple[Tensor]:
        """Returns pointcloud (1 point per occupied voxel).

        Returns:
            Tuple[Tensor]: points, features, weights, rgb, scale
        """
        return self._points, self._features, self._weights, self._rgb, self._scale

    def clone(self):
        """Deep copy of object. All internal tensors are cloned individually.

        Returns:
            VoxelizedPointcloud: New VoxelizedPointcloud object.
        """
        other = self.__class__({k: getattr(self, k) for k in self._INIT_ARGS})
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device: Union[str, torch.device]):
        """Move the object to a specific device.

        Args:
            device (Union[str, torch.device]): Device to move to.

        Returns:
            VoxelizedPointcloud: New VoxelizedPointcloud object on specified device.
        """
        other = self.clone()
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.to(device))
        return other

    def cpu(self):
        """Move the object to CPU.

        Returns:
            VoxelizedPointcloud: New VoxelizedPointcloud object on CPU.
        """
        return self.to("cpu")

    def cuda(self):
        """Move the object to CUDA.

        Returns:
            VoxelizedPointcloud: New VoxelizedPointcloud object on CUDA.
        """
        return self.to("cuda")

    def detach(self):
        """Detach object. All internal tensors are detached individually.

        Returns:
            VoxelizedPointcloud: New detached VoxelizedPointcloud object.
        """
        other = self.__class__({k: getattr(self, k) for k in self._INIT_ARGS})
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())
        return other


def voxelize(
    pos: Tensor,
    voxel_size: float,
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, Tensor]] = None,
    end: Optional[Union[float, Tensor]] = None,
) -> Tuple[Tensor]:
    """Returns voxel indices and packed (consecutive) indices for points

    Args:
        pos (Tensor): [N, 3] locations
        voxel_size (float): Size (resolution) of each voxel in the grid
        batch (Optional[Tensor], optional): Batch index of each point in pos. Defaults to None.
        start (Optional[Union[float, Tensor]], optional): Mins along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from min values in pos.
        end (Optional[Union[float, Tensor]], optional):  Maxes along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from max values in pos.
    Returns:
        voxel_idx (LongTensor): Idx of each point's voxel coordinate. E.g. [0, 0, 4, 3, 3, 4]
        cluster_consecutive_idx (LongTensor): Packed idx -- contiguous in cluster ID. E.g. [0, 0, 2, 1, 1, 2]
        batch_sample: See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/max_pool.html
    """
    voxel_cluster = voxel_grid(
        pos=pos, batch=batch, size=voxel_size, start=start, end=end
    )
    cluster_consecutive_idx, perm = consecutive_cluster(voxel_cluster)
    batch_sample = batch[perm] if batch is not None else None
    cluster_idx = voxel_cluster
    return cluster_idx, cluster_consecutive_idx, batch_sample


def scatter_weighted_mean(
    features: Tensor,
    weights: Tensor,
    cluster: Tensor,
    weights_cluster: Tensor,
    dim: int,
) -> Tensor:
    """_summary_

    Args:
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        cluster (LongTensor): [N] IDs of each point (clusters.max() should be <= N, or you'll OOM)
        weights_cluster (Tensor): [N,] aggregated weights of each cluster, used to normalize
        dim (int): Dimension along which to do the reduction -- should be 0

    Returns:
        Tensor: Agggregated features, weighted by weights and normalized by weights_cluster
    """
    assert dim == 0, "Dim != 0 not yet implemented"
    feature_cluster = scatter(
        features * weights[:, None], cluster, dim=dim, reduce="sum"
    )
    feature_cluster = feature_cluster / weights_cluster[:, None]
    return feature_cluster


def reduce_pointcloud(
    voxel_cluster: Tensor,
    pos: Tensor,
    features: Tensor,
    weights: Optional[Tensor] = None,
    rgbs: Optional[Tensor] = None,
    scales=Tensor,
    feature_reduce: str = "mean",
) -> Tuple[Tensor]:
    """Pools values within each voxel

    Args:
        voxel_cluster (LongTensor): [N] IDs of each point
        pos (Tensor): [N, 3] position of each point
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        rgbs (Optional[Tensor], optional): [N, 3] colors of each point. Defaults to None.
        feature_reduce (str, optional): Feature reduction method. Defaults to 'mean'.

    Raises:
        NotImplementedError: if unknown reduction method

    Returns:
        pos_cluster (Tensor): weighted average position within each voxel
        feature_cluster (Tensor): aggregated feature of each voxel
        weights_cluster (Tensor): aggregated weights of each voxel
        rgb_cluster (Tensor): colors of each voxel
    """
    if weights is None:
        weights = torch.ones_like(pos[..., 0])
    weights_cluster = scatter(weights, voxel_cluster, dim=0, reduce="sum")

    pos_cluster = scatter_weighted_mean(
        pos, weights, voxel_cluster, weights_cluster, dim=0
    )

    if rgbs is not None:
        rgb_cluster = scatter_weighted_mean(
            rgbs, weights, voxel_cluster, weights_cluster, dim=0
        )
    else:
        rgb_cluster = None

    if features is None:
        return pos_cluster, None, weights_cluster, rgb_cluster

    if feature_reduce == "mean":
        feature_cluster = scatter_weighted_mean(
            features, weights, voxel_cluster, weights_cluster, dim=0
        )
    elif feature_reduce == "max":
        feature_cluster = scatter(features, voxel_cluster, dim=0, reduce="max")
    elif feature_reduce == "sum":
        feature_cluster = scatter(
            features * weights[:, None], voxel_cluster, dim=0, reduce="sum"
        )
    else:
        raise NotImplementedError(f"Unknown feature reduction method {feature_reduce}")
    if scales is not None:
        scales = scales * weights.view(-1, 1)
        scales_out = scatter(scales, voxel_cluster, dim=0, reduce="sum") / weights_cluster.view(-1, 1)
    else:
        scales_out = None
    return pos_cluster, feature_cluster, weights_cluster, rgb_cluster, scales_out