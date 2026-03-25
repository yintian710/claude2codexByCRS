from fastapi import HTTPException, Request


MISSING_BEARER_DETAIL = "Missing upstream API key in Authorization header"


def get_bearer_api_key(raw_request: Request) -> str | None:
    authorization = raw_request.headers.get("authorization")
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None

    return token.strip()


def require_bearer_api_key(raw_request: Request) -> str:
    api_key = get_bearer_api_key(raw_request)
    if not api_key:
        raise HTTPException(status_code=502, detail=MISSING_BEARER_DETAIL)
    return api_key
