"""
Direct binary parser for ROS bag v2.0 files.
Bypasses the rosbag library entirely to handle truncated/unindexed bags.
Reference: http://wiki.ros.org/Bags/Format/2.0
"""
import struct
import os
import sys
import csv
import json
import bz2
import io
import numpy as np

BAG_PATH = "data/meadows_0.bag"
OUTPUT_DIR = "extracted"

# ROS bag record opcodes
OP_MSG_DEF    = 0x01  # Not standard, but sometimes used
OP_MSG_DATA   = 0x02
OP_BAG_HEADER = 0x03
OP_INDEX_DATA = 0x04
OP_CHUNK      = 0x05
OP_CHUNK_INFO = 0x06
OP_CONNECTION = 0x07


def read_uint32(f):
    data = f.read(4)
    if len(data) < 4:
        return None
    return struct.unpack("<I", data)[0]


def read_uint64(f):
    data = f.read(8)
    if len(data) < 8:
        return None
    return struct.unpack("<Q", data)[0]


def parse_header_fields(data):
    """Parse a ROS bag header into a dict of name->bytes."""
    fields = {}
    offset = 0
    while offset < len(data):
        if offset + 4 > len(data):
            break
        field_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if offset + field_len > len(data):
            break
        field = data[offset:offset + field_len]
        offset += field_len
        eq = field.find(b"=")
        if eq >= 0:
            name = field[:eq].decode("ascii", errors="replace")
            value = field[eq + 1:]
            fields[name] = value
    return fields


def parse_time(data):
    """Parse ROS time (secs, nsecs) from 8 bytes."""
    secs = struct.unpack_from("<I", data, 0)[0]
    nsecs = struct.unpack_from("<I", data, 4)[0]
    return secs, nsecs


class RosBagReader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.file_size = os.path.getsize(path)
        self.connections = {}  # conn_id -> {topic, datatype, md5sum, ...}
        
        # Read and verify header
        version_line = self.f.readline()
        if b"#ROSBAG V2.0" not in version_line:
            raise ValueError(f"Not a ROS bag v2.0 file: {version_line}")
        
        print(f"ROS bag v2.0, size: {self.file_size / 1e9:.2f} GB")
        
        # Read bag header record
        header_len = read_uint32(self.f)
        header_data = self.f.read(header_len)
        header = parse_header_fields(header_data)
        data_len = read_uint32(self.f)
        self.f.read(data_len)  # skip padding
        
        self.chunk_count = struct.unpack("<I", header["chunk_count"])[0]
        self.conn_count = struct.unpack("<I", header["conn_count"])[0]
        self.index_pos = struct.unpack("<Q", header["index_pos"])[0]
        
        print(f"Chunks: {self.chunk_count}, Connections: {self.conn_count}")
        print(f"Index pos: {self.index_pos} {'(TRUNCATED!)' if self.index_pos > self.file_size else '(OK)'}")
    
    def scan(self):
        """Scan through all records, yielding (topic, msg_data, timestamp) for message data records."""
        
        # Position after bag header
        self.f.seek(0)
        self.f.readline()  # skip version
        
        # Skip bag header
        hl = read_uint32(self.f)
        self.f.read(hl)
        dl = read_uint32(self.f)
        self.f.read(dl)
        
        chunk_num = 0
        msg_count = 0
        
        while self.f.tell() < self.file_size:
            pos = self.f.tell()
            
            # Read record header
            header_len = read_uint32(self.f)
            if header_len is None:
                break
            
            if header_len > 10_000_000:  # Sanity check
                print(f"  WARNING: Suspicious header length {header_len} at pos {pos}, stopping.")
                break
            
            header_data = self.f.read(header_len)
            if len(header_data) < header_len:
                print(f"  Truncated header at pos {pos}")
                break
            
            header = parse_header_fields(header_data)
            
            data_len = read_uint32(self.f)
            if data_len is None:
                break
            
            op = struct.unpack("<B", header.get("op", b"\x00"))[0] if "op" in header else -1
            
            if op == OP_CONNECTION:
                # Connection record - defines topic <-> conn_id mapping
                conn_data = self.f.read(data_len)
                conn_id = struct.unpack("<I", header["conn"])[0]
                topic = header.get("topic", b"").decode("ascii", errors="replace")
                
                # Parse connection data for message definition
                conn_fields = parse_header_fields(conn_data)
                datatype = conn_fields.get("type", b"unknown").decode("ascii", errors="replace")
                md5sum = conn_fields.get("md5sum", b"").decode("ascii", errors="replace")
                msg_def = conn_fields.get("message_definition", b"").decode("ascii", errors="replace")
                
                self.connections[conn_id] = {
                    "topic": topic,
                    "datatype": datatype,
                    "md5sum": md5sum,
                    "msg_def": msg_def,
                }
                
            elif op == OP_CHUNK:
                # Chunk record - contains compressed message data
                chunk_num += 1
                compression = header.get("compression", b"none").decode("ascii")
                chunk_size = struct.unpack("<I", header.get("size", b"\x00\x00\x00\x00"))[0]
                
                if chunk_num % 100 == 0:
                    pct = (self.f.tell() / self.file_size) * 100
                    print(f"  Chunk {chunk_num}/{self.chunk_count} ({pct:.0f}%), {msg_count} messages so far...")
                
                # Read chunk data
                chunk_data = self.f.read(data_len)
                if len(chunk_data) < data_len:
                    print(f"  Truncated chunk {chunk_num} at pos {pos}")
                    break
                
                # Decompress if needed
                if compression == "bz2":
                    try:
                        chunk_data = bz2.decompress(chunk_data)
                    except Exception as e:
                        print(f"  Failed to decompress chunk {chunk_num}: {e}")
                        continue
                elif compression == "lz4":
                    try:
                        import lz4.block
                        chunk_data = lz4.block.decompress(chunk_data, uncompressed_size=chunk_size)
                    except ImportError:
                        print("  WARNING: lz4 compression detected. Install: pip install lz4")
                        continue
                    except Exception as e:
                        print(f"  Failed to decompress lz4 chunk {chunk_num}: {e}")
                        continue
                elif compression != "none":
                    print(f"  Unknown compression: {compression}")
                    continue
                
                # Parse messages within the chunk
                chunk_io = io.BytesIO(chunk_data)
                while chunk_io.tell() < len(chunk_data):
                    # Read sub-record header
                    sub_hl_data = chunk_io.read(4)
                    if len(sub_hl_data) < 4:
                        break
                    sub_hl = struct.unpack("<I", sub_hl_data)[0]
                    
                    if sub_hl > 10_000_000:
                        break
                    
                    sub_header_data = chunk_io.read(sub_hl)
                    if len(sub_header_data) < sub_hl:
                        break
                    
                    sub_header = parse_header_fields(sub_header_data)
                    
                    sub_dl_data = chunk_io.read(4)
                    if len(sub_dl_data) < 4:
                        break
                    sub_dl = struct.unpack("<I", sub_dl_data)[0]
                    
                    sub_op = struct.unpack("<B", sub_header.get("op", b"\x00"))[0] if "op" in sub_header else -1
                    
                    if sub_op == OP_MSG_DATA:
                        # Message data record
                        conn_id = struct.unpack("<I", sub_header["conn"])[0]
                        time_secs, time_nsecs = parse_time(sub_header["time"])
                        
                        msg_data = chunk_io.read(sub_dl)
                        if len(msg_data) < sub_dl:
                            break
                        
                        msg_count += 1
                        
                        if conn_id in self.connections:
                            topic = self.connections[conn_id]["topic"]
                            datatype = self.connections[conn_id]["datatype"]
                            yield topic, datatype, msg_data, time_secs, time_nsecs
                    
                    elif sub_op == OP_CONNECTION:
                        # Connection record inside chunk
                        conn_data = chunk_io.read(sub_dl)
                        conn_id = struct.unpack("<I", sub_header["conn"])[0]
                        topic = sub_header.get("topic", b"").decode("ascii", errors="replace")
                        conn_fields = parse_header_fields(conn_data)
                        datatype = conn_fields.get("type", b"unknown").decode("ascii", errors="replace")
                        md5sum = conn_fields.get("md5sum", b"").decode("ascii", errors="replace")
                        msg_def = conn_fields.get("message_definition", b"").decode("ascii", errors="replace")
                        
                        self.connections[conn_id] = {
                            "topic": topic,
                            "datatype": datatype,
                            "md5sum": md5sum,
                            "msg_def": msg_def,
                        }
                    else:
                        # Skip unknown sub-records
                        chunk_io.read(sub_dl)
                
            elif op == OP_INDEX_DATA:
                # Index record - skip
                self.f.read(data_len)
                
            elif op == OP_CHUNK_INFO:
                # Chunk info - skip
                self.f.read(data_len)
                
            else:
                # Unknown record - skip
                if data_len > self.file_size:
                    print(f"  Suspicious data_len={data_len} at pos {pos}, stopping.")
                    break
                self.f.read(data_len)
        
        print(f"\nScan complete: {chunk_num} chunks, {msg_count} messages, {len(self.connections)} connections")
    
    def close(self):
        self.f.close()


# ─── ROS message deserializers ───────────────────────────────────────────────
def deserialize_ros_string(data, offset):
    """Read a ROS string (uint32 length + data)."""
    str_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    s = data[offset:offset + str_len].decode("ascii", errors="replace")
    offset += str_len
    return s, offset


def deserialize_image(data):
    """Deserialize sensor_msgs/Image."""
    offset = 0
    
    # Header: seq(uint32), stamp(uint32+uint32), frame_id(string)
    seq = struct.unpack_from("<I", data, offset)[0]; offset += 4
    stamp_secs = struct.unpack_from("<I", data, offset)[0]; offset += 4
    stamp_nsecs = struct.unpack_from("<I", data, offset)[0]; offset += 4
    frame_id, offset = deserialize_ros_string(data, offset)
    
    # Image fields
    height = struct.unpack_from("<I", data, offset)[0]; offset += 4
    width = struct.unpack_from("<I", data, offset)[0]; offset += 4
    encoding, offset = deserialize_ros_string(data, offset)
    is_bigendian = struct.unpack_from("<B", data, offset)[0]; offset += 1
    step = struct.unpack_from("<I", data, offset)[0]; offset += 4
    
    # Image data (uint8[])
    data_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
    img_data = data[offset:offset + data_len]
    
    return {
        "height": height,
        "width": width,
        "encoding": encoding,
        "step": step,
        "data": img_data,
        "stamp_secs": stamp_secs,
        "stamp_nsecs": stamp_nsecs,
    }


def deserialize_imu(data):
    """Deserialize sensor_msgs/Imu."""
    offset = 0
    
    # Header
    seq = struct.unpack_from("<I", data, offset)[0]; offset += 4
    stamp_secs = struct.unpack_from("<I", data, offset)[0]; offset += 4
    stamp_nsecs = struct.unpack_from("<I", data, offset)[0]; offset += 4
    frame_id, offset = deserialize_ros_string(data, offset)
    
    # Orientation (quaternion x,y,z,w)
    ox, oy, oz, ow = struct.unpack_from("<4d", data, offset); offset += 32
    # Orientation covariance (9 doubles)
    offset += 72
    
    # Angular velocity
    avx, avy, avz = struct.unpack_from("<3d", data, offset); offset += 24
    # Angular velocity covariance
    offset += 72
    
    # Linear acceleration
    lax, lay, laz = struct.unpack_from("<3d", data, offset); offset += 24
    
    return {
        "stamp_secs": stamp_secs, "stamp_nsecs": stamp_nsecs,
        "orientation": (ox, oy, oz, ow),
        "angular_velocity": (avx, avy, avz),
        "linear_acceleration": (lax, lay, laz),
    }


def deserialize_odometry(data):
    """Deserialize nav_msgs/Odometry."""
    offset = 0
    
    # Header
    seq = struct.unpack_from("<I", data, offset)[0]; offset += 4
    stamp_secs = struct.unpack_from("<I", data, offset)[0]; offset += 4
    stamp_nsecs = struct.unpack_from("<I", data, offset)[0]; offset += 4
    frame_id, offset = deserialize_ros_string(data, offset)
    
    # child_frame_id
    child_frame_id, offset = deserialize_ros_string(data, offset)
    
    # PoseWithCovariance: Pose (position + orientation) + float64[36]
    px, py, pz = struct.unpack_from("<3d", data, offset); offset += 24
    ox, oy, oz, ow = struct.unpack_from("<4d", data, offset); offset += 32
    offset += 288  # 36 * 8 covariance
    
    # TwistWithCovariance: Twist (linear + angular) + float64[36]
    lvx, lvy, lvz = struct.unpack_from("<3d", data, offset); offset += 24
    avx, avy, avz = struct.unpack_from("<3d", data, offset); offset += 24
    
    return {
        "stamp_secs": stamp_secs, "stamp_nsecs": stamp_nsecs,
        "position": (px, py, pz),
        "orientation": (ox, oy, oz, ow),
        "linear_velocity": (lvx, lvy, lvz),
        "angular_velocity": (avx, avy, avz),
    }


# ─── Main extraction ─────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    reader = RosBagReader(BAG_PATH)
    
    # Counters and file handles
    topic_counts = {}
    img_writers = {}   # topic -> (dir, csv_writer, csv_file, count)
    imu_writers = {}   # topic -> (csv_writer, csv_file, count)
    odom_writers = {}  # topic -> (csv_writer, csv_file, count)
    
    print("\nStarting extraction...\n")
    
    for topic, datatype, msg_data, t_secs, t_nsecs in reader.scan():
        if topic not in topic_counts:
            topic_counts[topic] = 0
        topic_counts[topic] += 1
        
        # ── Handle Image messages ──
        if "Image" in datatype and "image" in topic.lower():
            if topic not in img_writers:
                topic_dir = topic.strip("/").replace("/", "_")
                img_dir = os.path.join(OUTPUT_DIR, "images", topic_dir)
                os.makedirs(img_dir, exist_ok=True)
                ts_file = open(os.path.join(img_dir, "timestamps.csv"), "w", newline="")
                writer = csv.writer(ts_file)
                writer.writerow(["frame_index", "timestamp_sec", "timestamp_nsec", "filename", "encoding", "width", "height"])
                img_writers[topic] = {"dir": img_dir, "writer": writer, "file": ts_file, "count": 0}
                print(f"  Creating image output: {img_dir}")
            
            try:
                img = deserialize_image(msg_data)
                w = img_writers[topic]
                idx = w["count"]
                
                h, width, enc = img["height"], img["width"], img["encoding"]
                
                if enc in ("mono8", "8UC1"):
                    arr = np.frombuffer(img["data"], dtype=np.uint8).reshape(h, width)
                    fname = f"frame_{idx:06d}.png"
                    try:
                        from PIL import Image as PILImage
                        PILImage.fromarray(arr).save(os.path.join(w["dir"], fname))
                    except ImportError:
                        fname = f"frame_{idx:06d}.npy"
                        np.save(os.path.join(w["dir"], fname), arr)
                elif enc in ("rgb8",):
                    arr = np.frombuffer(img["data"], dtype=np.uint8).reshape(h, width, 3)
                    fname = f"frame_{idx:06d}.png"
                    try:
                        from PIL import Image as PILImage
                        PILImage.fromarray(arr).save(os.path.join(w["dir"], fname))
                    except ImportError:
                        fname = f"frame_{idx:06d}.npy"
                        np.save(os.path.join(w["dir"], fname), arr)
                elif enc in ("bgr8",):
                    arr = np.frombuffer(img["data"], dtype=np.uint8).reshape(h, width, 3)[:, :, ::-1]
                    fname = f"frame_{idx:06d}.png"
                    try:
                        from PIL import Image as PILImage
                        PILImage.fromarray(arr).save(os.path.join(w["dir"], fname))
                    except ImportError:
                        fname = f"frame_{idx:06d}.npy"
                        np.save(os.path.join(w["dir"], fname), arr)
                elif enc in ("32FC1",):
                    arr = np.frombuffer(img["data"], dtype=np.float32).reshape(h, width)
                    fname = f"frame_{idx:06d}.npy"
                    np.save(os.path.join(w["dir"], fname), arr)
                elif enc in ("16UC1", "mono16"):
                    arr = np.frombuffer(img["data"], dtype=np.uint16).reshape(h, width)
                    fname = f"frame_{idx:06d}.npy"
                    np.save(os.path.join(w["dir"], fname), arr)
                else:
                    fname = f"frame_{idx:06d}.npy"
                    np.save(os.path.join(w["dir"], fname), np.frombuffer(img["data"], dtype=np.uint8))
                
                w["writer"].writerow([idx, t_secs, t_nsecs, fname, enc, width, h])
                w["count"] += 1
                
                if w["count"] % 50 == 0:
                    print(f"    {topic}: {w['count']} frames")
                    
            except Exception as e:
                if img_writers[topic]["count"] < 2:
                    print(f"    WARNING: Failed to deserialize image from {topic}: {e}")
        
        # ── Handle IMU messages ──
        elif "Imu" in datatype:
            if topic not in imu_writers:
                imu_dir = os.path.join(OUTPUT_DIR, "imu")
                os.makedirs(imu_dir, exist_ok=True)
                topic_safe = topic.strip("/").replace("/", "_")
                csv_path = os.path.join(imu_dir, f"{topic_safe}.csv")
                f = open(csv_path, "w", newline="")
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_sec", "timestamp_nsec",
                    "orient_x", "orient_y", "orient_z", "orient_w",
                    "angular_vel_x", "angular_vel_y", "angular_vel_z",
                    "linear_acc_x", "linear_acc_y", "linear_acc_z",
                ])
                imu_writers[topic] = {"writer": writer, "file": f, "count": 0, "path": csv_path}
                print(f"  Creating IMU output: {csv_path}")
            
            try:
                imu = deserialize_imu(msg_data)
                w = imu_writers[topic]
                o = imu["orientation"]
                av = imu["angular_velocity"]
                la = imu["linear_acceleration"]
                w["writer"].writerow([t_secs, t_nsecs, o[0], o[1], o[2], o[3], av[0], av[1], av[2], la[0], la[1], la[2]])
                w["count"] += 1
                
                if w["count"] % 1000 == 0:
                    print(f"    {topic}: {w['count']} samples")
                    
            except Exception as e:
                if imu_writers[topic]["count"] == 0:
                    print(f"    WARNING: Failed to deserialize IMU from {topic}: {e}")
        
        # ── Handle Odometry messages ──
        elif "Odometry" in datatype:
            if topic not in odom_writers:
                odom_dir = os.path.join(OUTPUT_DIR, "odometry")
                os.makedirs(odom_dir, exist_ok=True)
                topic_safe = topic.strip("/").replace("/", "_")
                csv_path = os.path.join(odom_dir, f"{topic_safe}.csv")
                f = open(csv_path, "w", newline="")
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_sec", "timestamp_nsec",
                    "pos_x", "pos_y", "pos_z",
                    "orient_x", "orient_y", "orient_z", "orient_w",
                    "linear_vel_x", "linear_vel_y", "linear_vel_z",
                    "angular_vel_x", "angular_vel_y", "angular_vel_z",
                ])
                odom_writers[topic] = {"writer": writer, "file": f, "count": 0, "path": csv_path}
                print(f"  Creating odometry output: {csv_path}")
            
            try:
                odom = deserialize_odometry(msg_data)
                w = odom_writers[topic]
                p = odom["position"]
                o = odom["orientation"]
                lv = odom["linear_velocity"]
                av = odom["angular_velocity"]
                w["writer"].writerow([t_secs, t_nsecs, p[0], p[1], p[2], o[0], o[1], o[2], o[3], lv[0], lv[1], lv[2], av[0], av[1], av[2]])
                w["count"] += 1
                
                if w["count"] % 1000 == 0:
                    print(f"    {topic}: {w['count']} samples")
                    
            except Exception as e:
                if odom_writers[topic]["count"] == 0:
                    print(f"    WARNING: Failed to deserialize odometry from {topic}: {e}")
    
    # Close all files
    for w in img_writers.values():
        w["file"].close()
    for w in imu_writers.values():
        w["file"].close()
    for w in odom_writers.values():
        w["file"].close()
    
    reader.close()
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print("  EXTRACTION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n  All topics found:")
    for topic, count in sorted(topic_counts.items()):
        marker = ""
        if "image" in topic.lower():
            marker = " [IMAGE]"
        elif "imu" in topic.lower():
            marker = " [IMU]"
        elif "odom" in topic.lower():
            marker = " [ODOM]"
        elif "depth" in topic.lower() or "disparity" in topic.lower():
            marker = " [DEPTH]"
        print(f"    {topic:<55} {count:>6}{marker}")
    
    print(f"\n  Extracted:")
    for topic, w in img_writers.items():
        print(f"    {w['count']} images from {topic}")
    for topic, w in imu_writers.items():
        print(f"    {w['count']} IMU samples from {topic}")
    for topic, w in odom_writers.items():
        print(f"    {w['count']} odometry samples from {topic}")
    
    # Check for depth
    depth_topics = [t for t in topic_counts if "depth" in t.lower() or "disparity" in t.lower()]
    if depth_topics:
        print(f"\n  [OK] Depth/disparity topics found: {depth_topics}")
    else:
        print(f"\n  [X] No pre-computed depth/disparity topics.")
        print(f"    Compute stereo depth with OpenCV:")
        print(f"    focal=477.605px, baseline=0.2096m")
    
    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    
    # Save topic list
    with open(os.path.join(OUTPUT_DIR, "topics.json"), "w") as f:
        json.dump(topic_counts, f, indent=2)
    print(f"  Topic list saved to {OUTPUT_DIR}/topics.json")


if __name__ == "__main__":
    main()
