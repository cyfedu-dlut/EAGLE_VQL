"""Version information."""

__version__ = '1.0.0'
__version_info__ = tuple(int(x) for x in __version__.split('.'))

VERSION_MAJOR = __version_info__[0]
VERSION_MINOR = __version_info__[1]
VERSION_PATCH = __version_info__[2]

__status__ = 'stable'
__git_hash__ = 'unknown'

def get_version():
    """Get full version string."""
    version = __version__
    if __git_hash__ != 'unknown':
        version += f'+{__git_hash__[:7]}'
    return version

def get_version_info():
    """Get detailed version information."""
    import sys
    import torch
    
    return {
        'eagle_version': __version__,
        'git_hash': __git_hash__,
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
    }