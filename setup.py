from setuptools import setup

setup(
    name='easy_grad_cam',
    python_requires='>=3.5.2',
    version='1.0',
    description='A simple Grad-CAM tool for PyTorch.',
    author='enhuiz',
    author_email='niuzhe.nz@outlook.com',
    packages=['easy_grad_cam'],
    install_requires=['torch', 'opencv-python', 'numpy']
)
