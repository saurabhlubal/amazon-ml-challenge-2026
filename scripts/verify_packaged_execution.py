"""
Simulate exact SageMaker container execution of packaged code.
"""
import os
import sys
import shutil
import tarfile
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(PROJECT_ROOT, "test_sagemaker_sim")

def main():
    print("=" * 60)
    print("SIMULATING SAGEMAKER CONTAINER PACKAGING AND EXECUTION")
    print("=" * 60)
    
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(os.path.join(TEST_DIR, "code"), exist_ok=True)
    
    # 1. Package code
    tar_path = os.path.join(TEST_DIR, "sourcedir.tar.gz")
    print(f">> Packaging source code into {tar_path}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        for folder in ["business_entity_resolution", "sagemaker", "scripts"]:
            src = os.path.join(PROJECT_ROOT, folder)
            tar.add(src, arcname=folder)
            
    # 2. Extract into container /code directory
    code_dir = os.path.join(TEST_DIR, "code")
    print(f">> Extracting tarball into container simulation directory: {code_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(code_dir)
        
    # 3. Execute inside container simulation with mocked SageMaker SDK in sys.modules
    sim_script = """
import sys
import types

# Simulate AWS SageMaker SDK pre-installed in container
mock_sdk = types.ModuleType('sagemaker')
mock_sdk.__path__ = ['/usr/local/lib/python3.10/site-packages/sagemaker']
sys.modules['sagemaker'] = mock_sdk

# Execute entrypoint
import runpy
sys.argv = ['entrypoint.py', '--help']
runpy.run_path('sagemaker/entrypoint.py', run_name='__main__')
"""
    print(">> Executing entrypoint with simulated pre-installed AWS SageMaker SDK...")
    res = subprocess.run([sys.executable, "-c", sim_script], cwd=code_dir, capture_output=True, text=True)
    print(f"Exit Code: {res.returncode}")
    if res.returncode != 0:
        print(f"FAILED:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        sys.exit(1)
        
    print("STDOUT Preview:\n" + res.stdout[:300])
    print("\n[SUCCESS] Packaged container execution verified with zero import collisions!")
    
    # Clean up
    shutil.rmtree(TEST_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
