import torch

def cl_loss(cl_value, target_cl=1.0, convergence_threshold=(0.1, 2.0)):
    """
    Computes the loss for the current CL value, with a penalty for non-convergence.
    
    Args:
    cl_value (torch.Tensor): The current CL value from the CFD simulation.
    target_cl (float): The target CL value we want to achieve.
    convergence_threshold (tuple): The lower and upper bounds for CL considered as convergent.

    Returns:
    torch.Tensor: The computed loss.
    """
    # Check if the cl_value is within the convergence bounds
    convergent = (cl_value >= convergence_threshold[0]) & (cl_value <= convergence_threshold[1])

    # Calculate the base loss as the squared difference from the target CL
    loss = (cl_value - target_cl) ** 2

    # Apply a large penalty if the solution is not convergent
    # This can be adjusted to be more or less punitive
    penalty_factor = 10  # This is an arbitrary factor and should be tuned
    penalty = penalty_factor * torch.where(convergent, torch.tensor(0.0), torch.tensor(1.0))

    # Include the penalty in the total loss
    total_loss = loss + penalty

    return total_loss