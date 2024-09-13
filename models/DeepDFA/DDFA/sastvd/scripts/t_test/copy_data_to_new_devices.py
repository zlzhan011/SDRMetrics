import os
import shutil

def copy_directory(src, dst):
    """
    复制一个文件夹及其所有子文件夹和文件到目标文件夹
    :param src: 源文件夹路径
    :param dst: 目标文件夹路径
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def main():
    source_directory = '/data/cs_lzhan011/vulnerability'  # 源文件夹路径
    destination_directory = '/data3/3GPU/ODU/3GPU/data/cs_lzhan011'  # 目标文件夹路径

    # 复制目录
    copy_directory(source_directory, destination_directory)

if __name__ == "__main__":
    main()
