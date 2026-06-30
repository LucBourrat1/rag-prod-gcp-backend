import os
from typing import Optional

from fastapi import Header, HTTPException
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


async def verify_google_token(token: str) -> dict:
    try:
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        if idinfo["aud"] != GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=401, detail="Token invalide")
        return {
            "email": idinfo["email"],
            "name": idinfo.get("name", ""),
            "sub": idinfo["sub"],
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Token invalide: {str(e)}")


async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token manquant")
    token = authorization.replace("Bearer ", "")
    return await verify_google_token(token)
