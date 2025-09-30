import torch

def unique_each_row_vectorized(tensor, fill_value=0):
    """
    Returns a tensor containing unique elements in each row of the input tensor using a vectorized method,
    with positions not containing unique elements filled with `fill_value`.
    Args:
        tensor (torch.Tensor): Input tensor with shape (B, R, E), where
                               B is the batch size, R is the number of rows, and E is the number of elements per row.
        fill_value (int, float, ...) : Value to fill in duplicate positions. Default is 0.
    Returns:
        torch.Tensor: Tensor containing unique elements in each row, sorted in ascending order.
                      Positions without unique elements are filled with `fill_value`.
    """
    # Step 1: Sort the tensor along the last dimension
    sorted_tensor, _ = torch.sort(tensor, dim=-1)
    # Step 2: Create a mask to identify unique elements
    unique_mask = (sorted_tensor[..., 1:] != sorted_tensor[..., :-1])
    # Add a value of True at the first position since the first element is always unique in a sorted row
    unique_mask = torch.cat([torch.ones_like(unique_mask[..., :1], dtype=torch.bool, device=tensor.device), unique_mask], dim=-1)
    # Step 3: Apply the unique_mask to filter out duplicates
    unique_elements = sorted_tensor * unique_mask + (~unique_mask) * fill_value
 
    # Step 4: Sort again to move `fill_value` elements to the right side
    sorted_unique_elements, _ = torch.sort(unique_elements, dim=-1)
 
    return sorted_unique_elements

def create_table_from_index_and_value(index, value, fill_value=0.0):
    """
    Create a new table with the same shape as `index`, containing values from `value` based on `index`.
    Positions in `index` with -1 will be filled with `fill_value`.
    
    Args:
        index (torch.Tensor): Tensor containing indices, with shape [B, R, C].
        value (torch.Tensor): Tensor containing values, with shape [B, R, E].
        fill_value (float, int, ...): Value to fill in table cells where `index` is -1.
    
    Returns:
        torch.Tensor: A new table with the same shape as `index`, containing values from `value` or `fill_value`.
    """
    # Step 1: Create a mask for positions that are not -1
    mask = (index != -1)    
    # Step 2: Replace -1 values with 0 to ensure valid indices
    index_clipped = torch.clamp(index, min=0)
    # Step 3: Use torch.gather to fetch values from `value`
    gathered = torch.gather(value, dim=2, index=index_clipped)
    
    # Step 4: Create a result tensor with the default value
    output = torch.full_like(index_clipped, fill_value, dtype=value.dtype)
    
    # Step 5: Insert values from `gathered` into `output` at positions where mask == True
    output[mask] = gathered[mask]
    
    return output
