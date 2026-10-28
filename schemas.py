from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# --- CAMERA SCHEMAS ---

class CameraBase(BaseModel):
    name: str
    location: str
    url: str
    protocol: Optional[str] = None  # "HTTP" | "RTSP"

class CameraCreate(CameraBase):
    pass

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    protocol: Optional[str] = None
    is_active: Optional[bool] = None

class CameraStatusUpdate(BaseModel):
    status: str  # "ONLINE" | "OFFLINE"

class CameraResponse(BaseModel):
    id: str
    name: str
    location: str
    url: str
    protocol: str
    status: str
    is_active: bool
    last_seen: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- INCIDENT SCHEMAS ---

class IncidentBase(BaseModel):
    camera_id: str
    confidence: float
    camera_name: Optional[str] = None
    location: Optional[str] = None
    clip_duration_seconds: Optional[int] = 10

class IncidentResponse(BaseModel):
    id: str
    camera_id: str
    camera_name: str
    location: str
    confidence: float
    detected_at: datetime
    snapshot_url: str
    clip_url: Optional[str] = None
    clip_duration_seconds: Optional[int] = 10
    is_read: bool

    class Config:
        from_attributes = True


# --- LEGACY SCHEMAS ---

class AlertBase(BaseModel):
    device_id: int
    confidence: float

class AlertCreate(AlertBase):
    pass

class AlertResponse(BaseModel):
    id: int
    device_id: int
    confidence: float
    event_timestamp: datetime
    thumbnail_path: Optional[str] = None

    class Config:
        from_attributes = True
