# Multi-MCP Project

A complete solution for managing MCP servers with Ollama integration.

## Features

- **Web UI**: Modern interface for managing MCP servers
- **Multi-MCP Proxy**: Backend service for server management
- **Ollama Integration**: Local LLM support
- **Docker Compose**: Easy deployment
- **Monitoring**: Grafana and Prometheus integration

## Quick Start

1. **Prerequisites**:
   - Docker Desktop installed and running
   - Ollama installed locally (running on port 11434)

2. **Setup**:
   ```bash
   ./setup.sh
   ```

3. **Access Services**:
   - Frontend UI: http://localhost:3000
   - Backend API: http://localhost:3001
   - Grafana: http://localhost:3030 (admin/admin123)
   - Prometheus: http://localhost:9090

4. **Stop Services**:
   ```bash
   ./stop.sh
   ```

## Development

For development mode (with hot reload):
```bash
./dev.sh
```

## Configuration

Default MCP servers from your Claude Desktop config are automatically loaded:
- Grafana MCP Server
- Prometheus MCP Server

Add more servers through the web UI or modify the backend configuration.

## Project Structure

```
multi-mcp-project/
├── backend/           # Node.js backend server
├── frontend/          # Web UI
├── docker/           # Docker configurations
├── docker-compose.yml # Services orchestration
├── setup.sh          # Setup script
├── dev.sh            # Development script
└── stop.sh           # Stop script
```

## API Endpoints

- `GET /health` - Health check
- `GET /servers` - List all servers
- `POST /servers/:name` - Add new server
- `DELETE /servers/:name` - Remove server
- `POST /servers/:name/start` - Start server
- `POST /servers/:name/stop` - Stop server
- `POST /servers/:name/restart` - Restart server
- `GET /ollama/models` - List Ollama models
- `POST /ollama/generate` - Generate with Ollama

## Troubleshooting

1. **Docker issues**: Make sure Docker Desktop is running
2. **Port conflicts**: Check if ports 3000, 3001, 9090, 11434 are available
3. **Ollama connection**: Ensure Ollama is running on localhost:11434
4. **MCP servers**: Check Docker is available for MCP server containers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


docker-compose down
docker-compose build --no-cache multi-mcp-proxy
docker-compose up -d

# Watch the detailed logs
docker-compose logs -f multi-mcp-proxy