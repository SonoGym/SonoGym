import torch
import torch.nn.functional as F

def construct_lowest_y_array(label_map, label):
    """
    Construct a 2D array of size X * Z, where each cell contains the lowest y value 
    of the given label in a 3D label map.

    Parameters:
    - label_map: A 3D PyTorch tensor of shape (X, Y, Z) representing the label map.
    - label: Integer, the label to search for.

    Returns:
    - lowest_y_array: A 2D PyTorch tensor of shape (X, Z) where each cell contains the 
                      lowest y value for the given label or -1 if the label is not found.
    """
    # Get the dimensions of the label map
    X, Y, Z = label_map.shape

    # Initialize the output array with -1 (default for no label found)
    lowest_y_array = torch.full((X, Z), -1, dtype=torch.int64, device=label_map.device)

    # Iterate over each xz coordinate
    for x in range(X):
        for z in range(Z):
            # Get the y values where the label matches at this (x, z)
            y_indices = torch.nonzero(label_map[x, :, z] == label, as_tuple=False)

            if len(y_indices) > 0:
                # Find the smallest y index
                lowest_y = y_indices.min().item()
                lowest_y_array[x, z] = lowest_y

    return lowest_y_array


def apply_gaussian_smoothing(tensor, kernel_size=15, sigma=10.0):
    """
    Apply Gaussian smoothing to a 3D tensor.
    
    Parameters:
    - tensor: A 3D PyTorch tensor.
    - kernel_size: The size of the Gaussian kernel.
    - sigma: The standard deviation of the Gaussian kernel.
    
    Returns:
    - smoothed_tensor: The smoothed tensor.
    """
    # Create a Gaussian kernel
    coords = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()  # Normalize kernel

    # Create 3D separable kernel
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    kernel_3d = kernel_3d.to(tensor.device).unsqueeze(0).unsqueeze(0)

    # Apply convolution for smoothing
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    smoothed_tensor = F.conv3d(tensor, kernel_3d, padding=kernel_size // 2)
    return smoothed_tensor.squeeze()


def compute_boundary_normals(label_map, smoothing=True, kernel_size=15, sigma=10.0):
    """
    Compute the boundary normals for a 3D label map with optional smoothing.
    
    Parameters:
    - label_map: A 3D PyTorch tensor of shape (X, Y, Z).
    - smoothing: Whether to apply Gaussian smoothing to the gradients.
    - kernel_size: The size of the Gaussian kernel (if smoothing is enabled).
    - sigma: The standard deviation of the Gaussian kernel (if smoothing is enabled).
    
    Returns:
    - normals: A 4D tensor of shape (X, Y, Z, 3) representing smoothed normals.
    """
    # Compute gradients
    grad_x = label_map[2:, :, :] - label_map[:-2, :, :]
    grad_y = label_map[:, 2:, :] - label_map[:, :-2, :]
    grad_z = label_map[:, :, 2:] - label_map[:, :, :-2]

    # Pad gradients to match the original shape
    grad_x = F.pad(grad_x, (0, 0, 0, 0, 1, 1))
    grad_y = F.pad(grad_y, (0, 0, 1, 1, 0, 0))
    grad_z = F.pad(grad_z, (1, 1, 0, 0, 0, 0))

    # Apply Gaussian smoothing if enabled
    if smoothing:
        grad_x = apply_gaussian_smoothing(grad_x, kernel_size, sigma)
        grad_y = apply_gaussian_smoothing(grad_y, kernel_size, sigma)
        grad_z = apply_gaussian_smoothing(grad_z, kernel_size, sigma)

    # Normalize gradients
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
    grad_x /= grad_magnitude
    grad_y /= grad_magnitude
    grad_z /= grad_magnitude

    # Combine gradients into a normals tensor
    normals = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    return normals


def construct_boundary_normals_array(label_map, lowest_y_array, label, smoothing=True):
    """
    Construct a 3D array (X, Z, 3) where each cell contains the boundary normal at the 
    point with the lowest y value for the given label.
    
    Parameters:
    - label_map: A 3D PyTorch tensor of shape (X, Y, Z).
    - lowest_y_array: A 2D PyTorch tensor of shape (X, Z) containing the lowest y values for the label.
    - label: Integer, the label to compute normals for.
    - smoothing: Whether to apply smoothing to the normals.
    
    Returns:
    - normals_array: A 3D PyTorch tensor of shape (X, Z, 3) with smoothed boundary normals.
    """
    X, Y, Z = label_map.shape

    # Compute smoothed boundary normals
    normals = compute_boundary_normals((label_map == label).float(), smoothing)

    # Initialize normals array
    normals_array = torch.zeros((X, Z, 3), dtype=torch.float32, device=label_map.device)
    normals_array[:, :, 1] = 1.0

    x_grid, z_grid = torch.meshgrid(torch.arange(X), torch.arange(Z))

    lowest_ys = lowest_y_array[x_grid, z_grid]

    normals_array[x_grid, z_grid] = normals[x_grid, lowest_ys, z_grid]

    normal_norms = torch.linalg.norm(normals_array, dim=-1)
    inner_products = normals_array @ torch.tensor([0.0, 1.0, 0.0], device=label_map.device)
    normals_array[inner_products < 0, :] = - normals_array[inner_products < 0, :]
    normals_array[normal_norms < 1e-8, :] = torch.tensor([0.0, 1.0, 0.0], device=label_map.device)

    # Extract normals at the lowest y positions
    # for x in range(X):
    #     for z in range(Z):
    #         lowest_y = lowest_y_array[x, z]
    #         if lowest_y != -1:  # Valid boundary point
    #             if normals[x, lowest_y, z] @ torch.tensor([0.0, 1.0, 0.0], device=label_map.device) < 0:
    #                 normals[x, lowest_y, z] *= -1
    #             normals_array[x, z] = normals[x, lowest_y, z]
    #             if torch.linalg.norm(normals[x, lowest_y, z]) < 1e-8:
    #                 normals_array[x, z] = torch.tensor([0.0, 1.0, 0.0], device=label_map.device)

    return normals_array


def smooth_segmentation_labels(labels: torch.Tensor, num_classes: int, kernel_size: int = 3, sigma: float = 1.0):
    """
    Smooth boundaries of segmentation labels.
    
    Parameters:
        labels (torch.Tensor): Input label tensor with shape [H, W] or [B, H, W].
        num_classes (int): Total number of label classes.
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for the Gaussian kernel.
        
    Returns:
        torch.Tensor: Smoothed label tensor with shape [B, H, W] or [H, W].
    """
    # Ensure input is batch format
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)  # Add batch dimension
    
    B, H, W = labels.shape
    one_hot = F.one_hot(labels.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()  # Shape: [B, num_classes, H, W]

    # Create a Gaussian kernel
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=labels.device)
    gaussian = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gaussian /= gaussian.sum()
    gaussian_kernel = gaussian[:, None] @ gaussian[None, :]
    gaussian_kernel = gaussian_kernel.expand(num_classes, 1, -1, -1).to(labels.device)

    # Apply Gaussian smoothing
    padding = kernel_size // 2
    smoothed = F.conv2d(F.pad(one_hot, (padding, padding, padding, padding), mode='reflect'), 
                        gaussian_kernel, 
                        groups=num_classes)

    # Get final labels by taking the argmax
    smoothed_labels = torch.argmax(smoothed, dim=1)  # Shape: [B, H, W]
    
    return smoothed_labels.squeeze(0) if smoothed_labels.shape[0] == 1 else smoothed_labels