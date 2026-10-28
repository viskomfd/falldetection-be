from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=False)
    url = Column(Text, nullable=False)
    protocol = Column(String(16), nullable=False, default="HTTP")  # "HTTP" or "RTSP"
    status = Column(String(16), default="OFFLINE")                 # "ONLINE" or "OFFLINE"
    is_active = Column(Boolean, default=True)
    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    incidents = relationship("Incident", back_populates="camera", cascade="all, delete-orphan")

class Incident(Base):
    __tablename__ = "incidents"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    camera_id = Column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    camera_name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    snapshot_url = Column(Text, nullable=False)
    clip_url = Column(Text, nullable=True)
    clip_duration_seconds = Column(Integer, default=10)
    is_read = Column(Boolean, default=False)

    camera = relationship("Camera", back_populates="incidents")

# Legacy models kept for backward compatibility
class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    alerts = relationship("IncidentAlert", back_populates="device")

class IncidentAlert(Base):
    __tablename__ = "incident_alerts"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"))
    confidence = Column(Float, nullable=False)
    event_timestamp = Column(DateTime, default=datetime.utcnow)
    thumbnail_path = Column(Text, nullable=True)

    device = relationship("Device", back_populates="alerts")
