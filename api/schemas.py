from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    """A coordinate point"""
    lat: float
    lng: float


@dataclass
class MatchRequest:
    """Request to match routes"""
    location: Point
    shape: List[Point]


@dataclass
class Route:
    """A matched route"""
    id: str
    name: str
    distance: float  # miles
    duration: int  # minutes
    match_score: int  # percentage
    elevation_gain: int  # feet
    coordinates: List[Point]  # actual route path


@dataclass
class MatchResponse:
    """Response containing matched routes"""
    routes: List[Route]
    count: int
