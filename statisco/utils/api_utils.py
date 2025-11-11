from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from functools import wraps
import hashlib
import json

api_cache = {}

def cache_response(func):
    """
    decorator to cache API endpoint responses
    """
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        # unique key based on endpoint and query params
        cache_key = hashlib.sha256(f"{request.url.path}?{request.query_params}".encode()).hexdigest()

        if cache_key in api_cache:
            return JSONResponse(content=api_cache[cache_key])

        response = await func(request, *args, **kwargs)

        if isinstance(response, JSONResponse):
            api_cache[cache_key] = json.loads(response.body)

        return response
    return wrapper
