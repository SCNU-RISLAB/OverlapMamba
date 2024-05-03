import os


def create_folders_in_subfolders(folder_path, new_folder_name):
    # 获取文件夹路径下的所有子文件夹
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 在每个子文件夹中创建新文件夹（如果不存在同名文件夹）
    for subfolder in subfolders:
        new_folder_path = os.path.join(subfolder, new_folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created new folder: {new_folder_path}")
        else:
            print(f"Folder already exists, skipping: {new_folder_path}")


# 指定文件夹路径
folder_path = "/home/lenovo/xqc/OverlapTransformer-master/data_root_folder"

# 指定新文件夹名称
new_created_folder_name = "depth_map"


def rename_folders_in_subfolders(folder_path, old_folder_name, new_folder_name):
    # 获取文件夹路径下的所有子文件夹
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 遍历每个子文件夹，如果子文件夹包含名为old_folder_name的文件夹，则将其重命名为new_folder_name
    for subfolder in subfolders:
        folder_list = [f.name for f in os.scandir(subfolder) if f.is_dir()]
        if old_folder_name in folder_list:
            old_folder_path = os.path.join(subfolder, old_folder_name)
            new_folder_path = os.path.join(subfolder, new_folder_name)
            os.rename(old_folder_path, new_folder_path)
            print(f"Renamed folder from {old_folder_path} to {new_folder_path}")


# 指定要替换的旧文件夹名称和新文件夹名称
old_folder_name = "depth_map_50"
new_folder_name = "depth_map"

# 调用函数创建新文件夹
# create_folders_in_subfolders(folder_path, new_created_folder_name)

# 调用函数重命名文件夹
rename_folders_in_subfolders(folder_path, old_folder_name, new_folder_name)
