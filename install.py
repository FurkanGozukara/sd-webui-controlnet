import launch
from importlib import metadata
import sys
import os
import shutil
import platform
from pathlib import Path
from typing import Optional
from packaging.version import parse


repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"


def check_skimage_numpy_compatibility():
    """
    Test if scikit-image is compatible with the installed numpy.
    Returns True if compatible, False if there's a binary incompatibility.
    """
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", "from skimage._shared import geometry; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0 and "OK" in result.stdout
    except Exception:
        return True  # Assume OK if we can't test


def fix_scikit_image_numpy_compatibility():
    """
    Fix numpy/scikit-image binary incompatibility.
    
    When packages install a pre-built scikit-image wheel, it may be compiled 
    against numpy 2.x while the main WebUI requires numpy 1.x.
    This causes: ValueError: numpy.dtype size changed (Expected 96, got 88).
    
    Solution: Use scikit-image 0.19.0 which only has numpy 1.x compatible wheels.
    """
    print("sd-webui-controlnet: Checking scikit-image/numpy compatibility...")
    
    if check_skimage_numpy_compatibility():
        print("sd-webui-controlnet: scikit-image/numpy compatibility OK.")
        return
    
    print("sd-webui-controlnet: INCOMPATIBILITY DETECTED! Fixing...")
    print("sd-webui-controlnet: Installing scikit-image 0.19.0 (numpy 1.x compatible)...")
    
    import subprocess
    
    try:
        # Step 1: Uninstall scikit-image completely
        print("sd-webui-controlnet: Step 1/3 - Removing incompatible scikit-image...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "scikit-image", "-y"], check=False)
        
        # Step 2: Ensure numpy 1.26.0 is installed
        print("sd-webui-controlnet: Step 2/3 - Installing numpy 1.26.0...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy==1.26.0"],
            check=False
        )
        
        # Step 3: Install scikit-image 0.19.0 which only has numpy 1.x wheels
        print("sd-webui-controlnet: Step 3/3 - Installing scikit-image 0.19.0...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "scikit-image==0.19.0"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"pip output: {result.stdout}\n{result.stderr}")
            raise Exception("pip install failed")
        
        # Verify fix worked
        if check_skimage_numpy_compatibility():
            print("sd-webui-controlnet: SUCCESS! scikit-image/numpy compatibility fixed.")
        else:
            raise Exception("Fix did not resolve the incompatibility")
            
    except Exception as e:
        print(f"ERROR: Automatic fix failed: {e}")
        print("")
        print("=" * 60)
        print("MANUAL FIX REQUIRED:")
        print("=" * 60)
        print("Run these commands in your terminal:")
        print("")
        webui_root = repo_root.parent.parent
        if platform.system() == "Windows":
            print(f'  cd "{webui_root}"')
            print(f'  .\\venv\\Scripts\\pip.exe uninstall scikit-image -y')
            print(f'  .\\venv\\Scripts\\pip.exe install numpy==1.26.0 scikit-image==0.19.0')
        else:
            print(f'  cd "{webui_root}"')
            print(f'  ./venv/bin/pip uninstall scikit-image -y')
            print(f'  ./venv/bin/pip install numpy==1.26.0 scikit-image==0.19.0')
        print("")
        print("=" * 60)


def get_installed_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except Exception:
        return None


def extract_base_package(package_string: str) -> str:
    base_package = package_string.split("@git")[0]
    return base_package


def install_requirements(req_file):
    with open(req_file) as file:
        for package in file:
            try:
                package = package.strip()
                if "==" in package:
                    package_name, package_version = package.split("==")
                    installed_version = get_installed_version(package_name)
                    if installed_version != package_version:
                        launch.run_pip(
                            f'install -U "{package}"',
                            f"sd-webui-controlnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif ">=" in package:
                    package_name, package_version = package.split(">=")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or parse(
                        installed_version
                    ) < parse(package_version):
                        launch.run_pip(
                            f'install -U "{package}"',
                            f"sd-webui-controlnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif "<=" in package:
                    package_name, package_version = package.split("<=")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or parse(
                        installed_version
                    ) > parse(package_version):
                        launch.run_pip(
                            f'install "{package_name}=={package_version}"',
                            f"sd-webui-controlnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif not launch.is_installed(extract_base_package(package)):
                    launch.run_pip(
                        f'install "{package}"',
                        f"sd-webui-controlnet requirement: {package}",
                    )
            except Exception as e:
                print(e)
                print(
                    f"Warning: Failed to install {package}, some preprocessors may not work."
                )


def install_onnxruntime():
    """
    Install onnxruntime or onnxruntime-gpu based on the availability of CUDA.
    onnxruntime and onnxruntime-gpu can not be installed together.
    """
    if not launch.is_installed("onnxruntime") and not launch.is_installed("onnxruntime-gpu"):
        import torch.cuda as cuda # torch import head to improve loading time
        onnxruntime = 'onnxruntime-gpu' if cuda.is_available() else 'onnxruntime'
        launch.run_pip(
            f'install {onnxruntime}',
            f"sd-webui-controlnet requirement: {onnxruntime}",
        )


def try_install_from_wheel(pkg_name: str, wheel_url: str, version: Optional[str] = None):
    current_version = get_installed_version(pkg_name)
    if current_version is not None:
        # No version requirement.
        if version is None:
            return
        # Version requirement already satisfied.
        if parse(current_version) >= parse(version):
            return

    try:
        launch.run_pip(
            f"install --no-deps -U {wheel_url}",
            f"sd-webui-controlnet requirement: {pkg_name}",
        )
    except Exception as e:
        print(e)
        print(f"Warning: Failed to install {pkg_name}. Some processors will not work.")


def try_install_insight_face():
    """Attempt to install insightface library. The library is necessary to use ip-adapter faceid.
    Note: Building insightface library from source requires compiling C++ code, which should be avoided
    in principle. Here the solution is to download a precompiled wheel."""
    if get_installed_version("insightface") is not None:
        return

    default_win_wheel = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl"
    wheel_url = os.environ.get("INSIGHTFACE_WHEEL", default_win_wheel)

    system = platform.system().lower()
    architecture = platform.machine().lower()
    python_version = sys.version_info
    if wheel_url != default_win_wheel or (
        system == "windows"
        and "amd64" in architecture
        and python_version.major == 3
        and python_version.minor == 10
    ):
        try:
            launch.run_pip(
                f"install {wheel_url}",
                "sd-webui-controlnet requirement: insightface",
            )
        except Exception as e:
            print(e)
            print(
                "ControlNet init warning: Unable to install insightface automatically. "
            )
    else:
        print(
            "ControlNet init warning: Unable to install insightface automatically. "
            "Please try run `pip install insightface` manually."
        )


def try_remove_legacy_submodule():
    """Try remove annotators/hand_refiner_portable submodule dir."""
    submodule = repo_root / "annotator" / "hand_refiner_portable"
    if os.path.exists(submodule):
        try:
            shutil.rmtree(submodule)
        except Exception as e:
            print(e)
            print(
                f"Failed to remove submodule {submodule} automatically. You can manually delete the directory."
            )


install_requirements(main_req_file)
install_onnxruntime()
try_install_insight_face()
try_install_from_wheel(
    "handrefinerportable",
    wheel_url=os.environ.get(
        "HANDREFINER_WHEEL",
        "https://github.com/huchenlei/HandRefinerPortable/releases/download/v1.0.1/handrefinerportable-2024.2.12.0-py2.py3-none-any.whl",
    ),
    version="2024.2.12.0",
)

try_install_from_wheel(
    "depth_anything",
    wheel_url=os.environ.get(
        "DEPTH_ANYTHING_WHEEL",
        "https://github.com/huchenlei/Depth-Anything/releases/download/v1.0.0/depth_anything-2024.1.22.0-py2.py3-none-any.whl",
    ),
)

try_install_from_wheel(
    "depth_anything_v2",
    wheel_url=os.environ.get(
        "DEPTH_ANYTHING_V2_WHEEL",
        "https://github.com/MackinationsAi/UDAV2-ControlNet/releases/download/v1.0.0/depth_anything_v2-2024.7.1.0-py2.py3-none-any.whl",
    ),
)

try_install_from_wheel(
    "dsine",
    wheel_url=os.environ.get(
        "DSINE_WHEEL",
        "https://github.com/sdbds/DSINE/releases/download/1.0.2/dsine-2024.3.23-py3-none-any.whl",
    ),
)
try_remove_legacy_submodule()

# Fix scikit-image/numpy binary compatibility after all other packages are installed
# This must run LAST to ensure scikit-image is compatible with the installed numpy version
# fix_scikit_image_numpy_compatibility()
