from enum import Enum

class Status(str, Enum):
    NEW = "NEW"
    CHANGED = "CHANGED"
    UNCHANGED = "UNCHANGED"
    MISSING_SIDE = "MISSING_SIDE"
    DELETED = "DELETED"
    ORPHAN_VIDEO = "ORPHAN_VIDEO"
    PENDING = "PENDING"
    ERROR = "ERROR"
