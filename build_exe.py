"""
Windows EXE 打包脚本
在 Windows 上运行此脚本来生成 exe
"""

import subprocess
import sys
import os


def main():
    # 确保 PyInstaller 已安装
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 打包命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",  # 打包成单个 exe
        "--windowed",  # 不显示控制台窗口 (GUI 程序)
        # "--console",                  # 如果需要看调试信息，用这个替换上一行
        "--name", "RouletteDetector",  # exe 名称
        "--add-data", "roulette_config.json;." if os.name == 'nt' else "roulette_config.json:.",
        # 隐藏导入 - 解决某些库检测不到的问题
        "--hidden-import", "pynput.keyboard._win32",
        "--hidden-import", "pynput.mouse._win32",
        "--hidden-import", "cv2",
        "--hidden-import", "numpy",
        # 清理构建文件
        "--clean",
        # 主文件
        "roulette_detector.py"
    ]

    print("Running PyInstaller...")
    print(" ".join(cmd))
    subprocess.check_call(cmd)

    print("\n" + "=" * 50)
    print("Build complete!")
    print("EXE location: dist/RouletteDetector.exe")
    print("=" * 50)


if __name__ == "__main__":
    main()