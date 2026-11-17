
from .base import Locator, KvBuf, vma_name
from .litert_dmabuf import LitertDmabufLocator
from .litert_namechain import LitertNameChainLocator
from .litert_host import LitertHostLocator

LOCATORS = [LitertDmabufLocator(), LitertNameChainLocator(), LitertHostLocator()]

__all__ = ["Locator", "KvBuf", "vma_name", "LOCATORS",
           "LitertDmabufLocator", "LitertNameChainLocator", "LitertHostLocator"]
