import os
import glob
import numpy as np
from PIL import Image

# Configuration
# Using home expansion for safer paths
HOME = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data")
IMAGE_DIR_BASE = DATASET_DIR # Images are in subfolders here
DEPTH_DIR_BASE = os.path.join(HOME, "Downloads", "the_great_outdoors_data_depth_estimated") 
OUTPUT_DIR = os.path.join(HOME, "Downloads", "VTUGV_pointclouds_outdoors")
EXTRINSICS_DIR = os.path.join(DATASET_DIR, "camera2lidar", "camera2lidar")
INTRINSICS_DIR = os.path.join(DATASET_DIR, "camera_intrinsics", "camera_intrinsics")

CAMERAS = ["front_left", "front_right", "rear_center"]

def write_ply(path, points, colors):
    """
    Writes a PLY file (ASCII format) with vertex colors.
    points: (N, 3) float
    colors: (N, 3) float 0-1
    """
    num_points = len(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    
    with open(path, 'w') as f:
        f.write(header)
        # Vectorized string formatting for speed
        # Scale colors to 0-255
        colors_int = (colors * 255).clip(0, 255).astype(int)
        
        # Combine into one array for iteration
        data = np.hstack((points, colors_int))
        
        for row in data:
            f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} {int(row[3])} {int(row[4])} {int(row[5])}\n")

def parse_yaml_transform(path):
    """
    Parses a simple YAML file with structure:
    key:
      q:
        w: ...
        x: ...
        y: ...
        z: ...
      t:
        x: ...
        y: ...
        z: ...
    Returns (q, t) dictionaries.
    """
    import re
    q = {}
    t = {}
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("q:"):
                current_section = "q"
            elif line.startswith("t:"):
                current_section = "t"
            elif ":" in line:
                key, val = line.split(":")
                key = key.strip()
                val = val.strip()
                
                if current_section == "q" and key in ['w', 'x', 'y', 'z']:
                    q[key] = float(val)
                elif current_section == "t" and key in ['x', 'y', 'z']:
                    t[key] = float(val)
                    
    except Exception as e:
        print(f"Error parsing YAML {path}: {e}")
        return None, None
        
    return q, t

def quat_to_mat(q, t):
    """
    Convert quaternion (w,x,y,z) and translation (x,y,z) to 4x4 homogenous matrix.
    """
    w, x, y, z = q['w'], q['x'], q['y'], q['z']
    tx, ty, tz = t['x'], t['y'], t['z']
    
    # Rotation matrix from quaternion components
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

def load_extrinsics():
    """
    Loads extrinsics from YAML files.
    Returns dict: {cam_name: T_lidar_cam} (4x4 matrix transforming Camera -> LiDAR)
    
    Note: The YAML files provide T_lidar_from_camera (or Camera in LiDAR frame)?
    Actually, usually 'transformslid2left_cam' means 'transform from LiDAR frame to Left Camera frame'.
    i.e. P_cam = T_lid2cam * P_lid
    So if we have P_cam, we want P_lid.
    P_lid = inv(T_lid2cam) * P_cam.
    
    Let's assume filename 'transformslid2left_cam' -> T_c_l.
    We return T_l_c = inv(T_c_l).
    """
    extrinsics = {}
    mapping = {
        "front_left": "transformslid2left_cam.yaml",
        "front_right": "transformslid2right_cam.yaml",
        "rear_center": "transformslid2rear.yaml"
    }
    
    print("Loading extrinsics...")
    for cam, filename in mapping.items():
        path = os.path.join(EXTRINSICS_DIR, filename)
        if os.path.exists(path):
            q, t = parse_yaml_transform(path)
            if q and t:
                T_c_l = quat_to_mat(q, t)
                # We need T_l_c (Camera to LiDAR) to transform our points
                T_l_c = np.linalg.inv(T_c_l)
                extrinsics[cam] = T_l_c
                print(f"  Loaded {cam} extrinsics.")
            else:
                print(f"  Failed to parse {filename}")
        else:
            print(f"  Extrinsic file not found: {path}")
            
    return extrinsics

def get_rotation_z(degrees):
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

# Default Intrinsics
DEFAULT_INTRINSICS = (300.0, 300.0, 320.0, 240.0) # fx, fy, cx, cy

def load_intrinsics():
    intrinsics = {}
    mapping = {
        "front_left": "left_cam_intrinsic.txt",
        "front_right": "right_cam_intrinsic.txt",
        "rear_center": "rear_cam_intrinsic.txt"
    }
    
    for cam, filename in mapping.items():
        path = os.path.join(INTRINSICS_DIR, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    line = f.readline().strip()
                    parts = line.split()
                    if len(parts) >= 4:
                        fx, fy, cx, cy = map(float, parts[:4])
                        intrinsics[cam] = (fx, fy, cx, cy)
                        print(f"Loaded intrinsics for {cam}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                    else:
                        print(f"Invalid format in {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Intrinsic file not found: {path} (Using defaults)")
            
    return intrinsics

IMU_CSV_PATH = os.path.join(DATASET_DIR, "imu_data.csv")

def load_imu_data():
    """
    Parses imu_data.csv into numpy arrays.
    Returns dict with:
      timestamps: (N,) int64 nanoseconds
      orientations: (N, 4) float64 [w, x, y, z]
      accelerations: (N, 3) float64 [ax, ay, az]
    """
    print("Loading IMU data...")
    timestamps = []
    orientations = []  # w, x, y, z
    accelerations = []  # ax, ay, az
    
    with open(IMU_CSV_PATH, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 12:
                continue
            ts = int(parts[0])  # bag_timestamp
            qx, qy, qz, qw = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            ax, ay, az = float(parts[9]), float(parts[10]), float(parts[11])
            
            timestamps.append(ts)
            orientations.append([qw, qx, qy, qz])
            accelerations.append([ax, ay, az])
    
    result = {
        'timestamps': np.array(timestamps, dtype=np.int64),
        'orientations': np.array(orientations, dtype=np.float64),
        'accelerations': np.array(accelerations, dtype=np.float64),
    }
    print(f"  Loaded {len(timestamps)} IMU readings.")
    print(f"  Time span: {(timestamps[-1] - timestamps[0]) / 1e9:.1f} seconds")
    return result

def quat_to_rotation_array(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    Works with single quaternion as 1D array.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ])

def dead_reckon_positions(imu_data, subsample=10):
    """
    Estimate positions by double-integrating gravity-compensated acceleration.
    Uses every `subsample`-th IMU reading for speed.
    Returns:
      dr_timestamps: (M,) int64
      dr_positions: (M, 3) float64  world-frame positions
      dr_orientations: (M, 4) float64  [w,x,y,z] quaternions
    """
    ts_all = imu_data['timestamps']
    q_all = imu_data['orientations']
    a_all = imu_data['accelerations']
    
    # Subsample for speed
    indices = np.arange(0, len(ts_all), subsample)
    ts = ts_all[indices]
    qs = q_all[indices]
    accs = a_all[indices]
    
    gravity_world = np.array([0.0, 0.0, -9.81])  # Gravity in world frame (z-up)
    
    positions = np.zeros((len(ts), 3))
    velocity = np.zeros(3)
    
    for i in range(1, len(ts)):
        dt = (ts[i] - ts[i-1]) / 1e9  # nanoseconds to seconds
        if dt <= 0 or dt > 0.5:  # Skip bad gaps
            continue
        
        # Rotate acceleration from body to world frame
        R = quat_to_rotation_array(qs[i])
        acc_world = R @ accs[i]
        
        # Remove gravity
        acc_world_no_g = acc_world - gravity_world
        
        # Simple Euler integration
        velocity += acc_world_no_g * dt
        positions[i] = positions[i-1] + velocity * dt
    
    print(f"  Dead reckoning: {len(ts)} poses, "
          f"total displacement: {np.linalg.norm(positions[-1] - positions[0]):.1f}m")
    
    return ts, positions, qs

def get_world_pose_at_timestamp(target_ts, dr_timestamps, dr_positions, dr_orientations):
    """
    Get the 4x4 world pose at a given timestamp by interpolating the dead-reckoned trajectory.
    """
    idx = np.searchsorted(dr_timestamps, target_ts)
    idx = np.clip(idx, 0, len(dr_timestamps) - 1)
    
    # Get closest orientation and position
    q = dr_orientations[idx]
    pos = dr_positions[idx]
    
    R = quat_to_rotation_array(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

def get_timestamp_from_filename(filename):
    name = os.path.splitext(filename)[0]
    if name.isdigit():
        return int(name)
    return None

def find_closest_matches(cam_files_dict, tolerance_ns=50000000): # 50ms tolerance
    ref_cam = "front_left"
    if ref_cam not in cam_files_dict:
        if not cam_files_dict: return []
        ref_cam = list(cam_files_dict.keys())[0]
    
    other_cams = [c for c in CAMERAS if c != ref_cam]
    matches = [] 
    
    # Get Valid Timestamps for Ref
    ref_timestamps = []
    for f in cam_files_dict[ref_cam].keys():
        ts = get_timestamp_from_filename(f)
        if ts is not None:
            ref_timestamps.append(ts)
    ref_timestamps.sort()
    
    # Get Sorted Arrays for Others
    other_ts_data = {}
    for cam in other_cams:
        if cam in cam_files_dict:
            ts_list = []
            for f in cam_files_dict[cam].keys():
                ts = get_timestamp_from_filename(f)
                if ts is not None:
                    ts_list.append(ts)
            ts_list.sort()
            other_ts_data[cam] = np.array(ts_list)
        else:
            other_ts_data[cam] = np.array([])

    print(f"Matching frames using {ref_cam} as reference ({len(ref_timestamps)} frames)...")
    
    for t_ref in ref_timestamps:
        match_entry = {ref_cam: str(t_ref)}
        valid_set = True
        
        for cam in other_cams:
            ts_arr = other_ts_data[cam]
            if len(ts_arr) == 0:
                valid_set = False; break
                
            idx = np.searchsorted(ts_arr, t_ref)
            
            candidates = []
            if idx < len(ts_arr): candidates.append(ts_arr[idx])
            if idx > 0: candidates.append(ts_arr[idx-1])
                
            if not candidates:
                valid_set = False; break
                
            closest_t = min(candidates, key=lambda x: abs(x - t_ref))
            diff = abs(closest_t - t_ref)
            
            if diff <= tolerance_ns:
                match_entry[cam] = str(closest_t)
            else:
                valid_set = False; break
        
        if valid_set:
            matches.append(match_entry)
            
    return matches

def main():
    # ===== CONFIGURATION =====
    WINDOW_SIZE = 50      # Number of frames to accumulate per PLY
    SUBSAMPLE_RATE = 0.03 # Keep 3% of pixels per frame (lower because we accumulate)
    DEPTH_MIN_M = 0.5
    DEPTH_MAX_M = 50.0
    MAX_RENDER_DEPTH = 40.0
    # =========================

    if not os.path.exists(DEPTH_DIR_BASE):
        print(f"Depth directory not found: {DEPTH_DIR_BASE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cam_intrinsics = load_intrinsics()
    cam_extrinsics = load_extrinsics()

    # Load IMU and compute dead-reckoned trajectory
    imu_data = load_imu_data()
    dr_timestamps, dr_positions, dr_orientations = dead_reckon_positions(imu_data, subsample=10)

    # 1. Gather all files
    cam_files = {}
    for cam in CAMERAS:
        img_dir = os.path.join(IMAGE_DIR_BASE, cam)
        if not os.path.exists(img_dir):
            print(f"Warning: Camera folder {cam} not found")
            continue
            
        files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        files.sort()
        cam_files[cam] = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
        print(f"Found {len(cam_files[cam])} images for {cam}")

    if not cam_files:
        print("No images found.")
        return
        
    # 2. Match tuples
    matched_frames = find_closest_matches(cam_files, tolerance_ns=50000000)
    print(f"Found {len(matched_frames)} synchronized frame tuples.")
    
    if not matched_frames:
        print("No matched frames found.")
        return

    # 3. Process in Windows
    num_windows = (len(matched_frames) + WINDOW_SIZE - 1) // WINDOW_SIZE
    print(f"Processing {len(matched_frames)} frames in {num_windows} windows of {WINDOW_SIZE}...")
    
    for win_idx in range(num_windows):
        start = win_idx * WINDOW_SIZE
        end = min(start + WINDOW_SIZE, len(matched_frames))
        window_frames = matched_frames[start:end]
        
        # Use first frame's ref timestamp for the window name
        win_ref_name = window_frames[0].get("front_left", f"win_{win_idx}")
        print(f"\n=== Window {win_idx+1}/{num_windows} (frames {start}-{end-1}, ref={win_ref_name}) ===")
        
        # Get the world pose of the FIRST frame in this window as the reference origin
        first_ts = int(window_frames[0].get("front_left", "0"))
        T_world_ref = get_world_pose_at_timestamp(first_ts, dr_timestamps, dr_positions, dr_orientations)
        T_ref_world = np.linalg.inv(T_world_ref)  # We'll express everything relative to window start
        
        window_points = []
        window_colors = []
        
        for i, match in enumerate(window_frames):
            # Get the world pose for this frame
            frame_ts = int(match.get("front_left", "0"))
            T_world_frame = get_world_pose_at_timestamp(frame_ts, dr_timestamps, dr_positions, dr_orientations)
            
            # Transform: world -> relative to window origin
            T_relative = T_ref_world @ T_world_frame
            
            for cam in CAMERAS:
                if cam not in match:
                    continue
                basename = match[cam]
                if basename not in cam_files.get(cam, {}):
                    continue
                img_path = cam_files[cam][basename]
                depth_path = os.path.join(DEPTH_DIR_BASE, cam, basename + ".npy")
                
                if not os.path.exists(depth_path):
                    continue
                    
                try:
                    rgb_img = Image.open(img_path).convert("RGB")
                    rgb_arr = np.array(rgb_img)
                    
                    depth = np.load(depth_path)
                    if depth.ndim == 3:
                        depth = depth.squeeze()
                    
                    # Resize depth if needed
                    if depth.shape[:2] != rgb_arr.shape[:2]:
                        depth_pil = Image.fromarray(depth)
                        depth_pil = depth_pil.resize(rgb_arr.shape[:2][::-1], Image.NEAREST)
                        depth = np.array(depth_pil)

                    # Depth normalization (relative 0-255 -> metric meters)
                    depth_inv = 255.0 - depth
                    depth_norm = depth_inv / 255.0
                    depth_metric = DEPTH_MIN_M + depth_norm * (DEPTH_MAX_M - DEPTH_MIN_M)

                    H, W = depth_metric.shape
                    
                    # Subsample
                    keep_mask = np.random.rand(H, W) < SUBSAMPLE_RATE
                    valid_mask = (depth_metric > DEPTH_MIN_M) & (depth_metric < MAX_RENDER_DEPTH) & keep_mask
                    
                    y_idxs, x_idxs = np.where(valid_mask)
                    z_vals = depth_metric[valid_mask]
                    colors = rgb_arr[valid_mask] / 255.0
                    
                    if len(z_vals) == 0:
                        continue
                    
                    # Intrinsics
                    if cam in cam_intrinsics:
                        fx, fy, cx, cy = cam_intrinsics[cam]
                    else:
                        fx, fy, cx, cy = DEFAULT_INTRINSICS
                    
                    # Backproject to camera frame
                    x = (x_idxs - cx) * z_vals / fx
                    y = (y_idxs - cy) * z_vals / fy
                    points_cam = np.stack([x, y, z_vals], axis=1)
                    
                    # Camera -> Vehicle/LiDAR frame
                    if cam in cam_extrinsics:
                        T_veh_cam = cam_extrinsics[cam]
                    else:
                        T_veh_cam = np.eye(4)
                    
                    points_cam_h = np.column_stack((points_cam, np.ones(len(points_cam))))
                    points_veh = (T_veh_cam @ points_cam_h.T).T[:, :3]
                    
                    # Vehicle frame -> World frame (relative to window origin)
                    points_veh_h = np.column_stack((points_veh, np.ones(len(points_veh))))
                    points_world = (T_relative @ points_veh_h.T).T[:, :3]
                    
                    window_points.append(points_world)
                    window_colors.append(colors)
                    
                except Exception as e:
                    print(f"  Error {cam}/{basename}: {e}")
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(window_frames)} frames in window...")
                    
        if window_points:
            all_points = np.vstack(window_points)
            all_colors = np.vstack(window_colors)
            
            save_path = os.path.join(OUTPUT_DIR, f"accumulated_{win_ref_name}_w{WINDOW_SIZE}.ply")
            write_ply(save_path, all_points, all_colors)
            print(f"  Saved {save_path} ({len(all_points)} points)")
        else:
            print(f"  Window {win_idx+1} produced no points.")
            
    print("\nTemporal fusion complete.")

if __name__ == "__main__":
    main()

