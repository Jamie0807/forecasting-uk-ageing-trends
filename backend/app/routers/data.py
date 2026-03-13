from fastapi import APIRouter
from app.services.data_service import get_ageing_ratio, get_regions, get_overview_stats

router = APIRouter()


@router.get("/ageing-ratio")
def ageing_ratio():
    """Return historical 65+ ageing ratio per region and year."""
    return get_ageing_ratio()


@router.get("/regions")
def regions():
    """Return list of available regions."""
    return get_regions()


@router.get("/overview")
def overview():
    """Return summary statistics per region."""
    return get_overview_stats()
