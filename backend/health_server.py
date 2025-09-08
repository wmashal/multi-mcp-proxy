#!/usr/bin/env python3
"""
Simple health check server that runs alongside multi-mcp
"""
import asyncio
from aiohttp import web

async def health(request):
    return web.json_response({"status": "healthy", "service": "multi-mcp"})

async def create_app():
    app = web.Application()
    app.router.add_get('/health', health)
    return app

if __name__ == '__main__':
    app = asyncio.run(create_app())
    web.run_app(app, host='0.0.0.0', port=8082)