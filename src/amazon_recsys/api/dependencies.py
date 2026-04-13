from fastapi import Depends, Request

from amazon_recsys.application.services import BundleRecommendationService
from amazon_recsys.config.container import Container
from amazon_recsys.config.settings import AppSettings


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_recommendation_service(container: Container = Depends(get_container)) -> BundleRecommendationService:
    return container.recommendation_service


def get_settings(container: Container = Depends(get_container)) -> AppSettings:
    return container.settings
