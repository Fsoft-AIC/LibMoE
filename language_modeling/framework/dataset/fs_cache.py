import os
import psutil
from collections import namedtuple
from typing import List, Optional, Dict
import getpass
import re
from tqdm import tqdm
import ctypes
import sys
# from ..utils.lockfile import LockFile

is_linux = sys.platform.startswith('linux')

cache_paths = []
cached_files = {}
cache_lock_path = None


def max_file_size(directory):
    try:
        statvfs = os.statvfs(directory)
        available_space = min(statvfs.f_frsize * statvfs.f_bavail, statvfs.f_frsize * statvfs.f_blocks)
        return available_space
    except FileNotFoundError:
        return None


DiskInfo = namedtuple('DiskInfo', ['mount_point', 'total_space', 'disk_type'])


def has_write_permission(directory: str) -> bool:
    return os.access(directory, os.W_OK)


def get_disk_type(disk: str) -> str:
    try:
        partitions = psutil.disk_partitions(all=False)
        for partition in partitions:
            if partition.mountpoint == disk:
                if psutil.disk_io_counters(perdisk=True).get(partition.device, None):
                    return "HDD"
                else:
                    return "SSD"
    except PermissionError:
        return "Unknown"


def get_space_for_disks(disks: List[DiskInfo]) -> Dict[str, str]:
    return {dsk.mount_point: max_file_size(dsk.mount_point) for dsk in disks}


def sort_disks(disks: List[DiskInfo]) -> List[DiskInfo]:
    spaces = get_space_for_disks(disks)
    disks.sort(key=lambda x: (x.disk_type == 'SSD',  spaces[x.mount_point]), reverse=True)
    return disks


def get_local_disks() -> List[DiskInfo]:
    disk_info: List[DiskInfo] = []
    for partition in psutil.disk_partitions(all=False):
        if partition.mountpoint:
            if has_write_permission(partition.mountpoint):
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_type = get_disk_type(partition.mountpoint)
                    disk_info.append(DiskInfo(partition.mountpoint, usage.total, disk_type))
                except PermissionError:
                    continue

    return sort_disks(disk_info)


def get_username() -> str:
    return getpass.getuser()

def init_fs_cache(pattern: Optional[str] = None):
    global cache_paths
    global cached_files
    global cache_lock_path

    if pattern is None or not is_linux:
        cache_paths = []
        cached_files = {}
        return

    if pattern == "*":
        pattern = ".*"

    pattern = re.compile(pattern)
    disks = get_local_disks()
    for disk in disks:
        if pattern.match(disk.mount_point):
            cache_paths.append(disk)

    if len(cache_paths) > 0:
        d1 = list(sorted([c.mount_point for c in cache_paths]))[0]
        cache_lock_path = f"{d1}/{get_username()}/fs_cache.lock"
        os.makedirs(os.path.dirname(cache_lock_path), exist_ok=True)


def is_cached_file_valid(original_file_path: str, cached_file_path: str) -> bool:
    # Check if both files exist
    if not (os.path.exists(original_file_path) and os.path.exists(cached_file_path)):
        return False

    # Get information about the original file
    original_stat = os.stat(original_file_path)
    original_size = original_stat.st_size
    original_last_modified = original_stat.st_mtime

    # Get information about the cached file
    cached_stat = os.stat(cached_file_path)
    cached_size = cached_stat.st_size
    cached_last_modified = cached_stat.st_mtime

    # Check if the sizes match and the cached version is newer
    return original_size == cached_size and original_last_modified < cached_last_modified


def copy_file(source: str, destination: str) -> None:
    total_size = os.path.getsize(source)
    chunk_size = 1024 * 1024 * 10

    try:
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(source, 'rb') as fsrc:
                with open(destination, 'wb') as fdst:
                    while True:
                        buf = fsrc.read(chunk_size)
                        if not buf:
                            break
                        fdst.write(buf)
                        pbar.update(len(buf))
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise e


class StatFS(ctypes.Structure):
    _fields_ = [
        ("f_type", ctypes.c_long),
        ("idontcare", ctypes.c_long * 128)
    ]

statfs = None

def load_statfs():
    global statfs
    if statfs is not None:
        return

    # Load libc
    libc = ctypes.CDLL(None)

    # Define statfs function
    statfs = libc.statfs
    statfs.argtypes = [ctypes.c_char_p, ctypes.POINTER(StatFS)]
    statfs.restype = ctypes.c_int

def get_fs_type(path: str) -> str:
    load_statfs()
    buf = StatFS()
    result = statfs(path.encode(), ctypes.pointer(buf))
    if result == 0:
        return buf.f_type
    else:
        return 0

def is_nfs(path: str) -> bool:
    return get_fs_type(path) in {0x6969, 0x517B}

def get_cached_file(file_path: str) -> str:
    global cache_paths
    global cached_files
    global cache_lock_path

    if not cache_paths:
        return file_path

    file_path_full = os.path.realpath(file_path)

    if file_path in cached_files:
        return cached_files[file_path_full]

    if not is_nfs(file_path_full):
        # Cache only network files.
        return file_path

    username = get_username()

    # with LockFile(cache_lock_path):
    for disk in cache_paths:
        cached_path = f"{disk.mount_point}/{username}/fs_cache/{file_path_full}"

        if os.path.isfile(cached_path):
            if not is_cached_file_valid(file_path_full, cached_path):
                print(f"CACHE: Cache file invalid: {cached_path}. Copying file to cache...")
                os.remove(cached_path)
                try:
                    copy_file(file_path_full, cached_path)
                except Exception as e:
                    print(e)
                    print(f"CACHE: Failed to copy file to cache: {cached_path}. Ignoring...")
                    continue

            cached_files[file_path_full] = cached_path
            return cached_path

        # If no cache path was found, allocate it on the disk with the most space (the first in the list that has
        # enough space)
        cache_paths = sort_disks(cache_paths)
        available_space = get_space_for_disks(cache_paths)

        for path in cache_paths:
            if available_space[path.mount_point] > os.path.getsize(file_path_full):
                cached_path = f"{disk.mount_point}/{username}/fs_cache/{file_path_full}"
                try:
                    print(f"CACHE: Copying new file {file_path_full} to cache...")
                    os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                    copy_file(file_path_full, cached_path)
                except Exception as e:
                    print(e)
                    print(f"CACHE: Failed to copy file to cache: {cached_path}. Ignoring...")
                    continue

                cached_files[file_path_full] = cached_path
                return cached_path

        print(f"CACHE: Failed to find a cache path with enough space for {file_path}. Using original file.")
        return file_path
