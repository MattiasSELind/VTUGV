import torch
from queryocc_components import QueryOccNet

def test_queryocc():
    B = 2
    H, W = 272, 512 # Reduced for faster CPU test
    
    # Mock Inputs
    img_prev = torch.randn(B, 3, H, W)
    img_curr = torch.randn(B, 3, H, W)
    img_next = torch.randn(B, 3, H, W)
    images = [img_prev, img_curr, img_next]
    
    # Mock Poses (Identity for simplicity)
    pose = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
    poses = [pose, pose, pose]
    
    # Mock Intrinsics
    K = torch.tensor([[500, 0, W/2], [0, 500, H/2], [0, 0, 1]]).unsqueeze(0).expand(B, -1, -1)
    
    # Mock Extrinsics
    T_c2v = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
    
    model = QueryOccNet(img_size=(H, W))
    
    print("Running forward pass...")
    try:
        depth, rgb = model(images, poses, K, T_c2v)
        print(f"Success!")
        print(f"Depth shape: {depth.shape}")
        print(f"RGB shape:   {rgb.shape}")
        
        # Check if upsampling is needed
        if depth.shape[-2:] != (H, W):
            print(f"MISMATCH: Expected H,W={(H,W)}, got {depth.shape[-2:]}")
        else:
            print("Resolution matches input.")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_queryocc()
