from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer_scheme = HTTPBearer(auto_error=False)


def get_expected_api_token() -> str:
    return os.getenv("FISHFARM_API_TOKEN", "dev-token")


def get_expected_username() -> str:
    return os.getenv("FISHFARM_USERNAME", "admin")


def get_expected_password() -> str:
    return os.getenv("FISHFARM_PASSWORD", "admin123")


def require_api_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> str:
    expected = get_expected_api_token()
    if credentials is None or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    return credentials.credentials
