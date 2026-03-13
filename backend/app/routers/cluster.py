from fastapi import APIRouter, Query
from app.services.cluster_service import get_cluster_results

router = APIRouter()


@router.get("/cluster")
def cluster(n_clusters: int = Query(3, ge=2, le=10, description="Number of clusters")):
    """Return KMeans cluster assignments and trend data per region."""
    return get_cluster_results(n_clusters=n_clusters)
