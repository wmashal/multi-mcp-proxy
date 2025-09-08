# Multi-MCP Proxy

A comprehensive Model Context Protocol (MCP) proxy system that provides a web interface for managing multiple MCP servers through the [multi-mcp](https://github.com/kfirtoledo/multi-mcp) server, integrated with Azure OpenAI for intelligent tool orchestration.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Frontend      │    │    Backend       │    │  Multi-MCP Server   │
│   (Web UI)      │◄──►│   (FastAPI)      │◄──►│  (GitHub Repo)      │
│   Port: 3001    │    │   Port: 8001     │    │   Port: 8081        │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌──────────────────┐    ┌──────────────────┐
                       │   Azure OpenAI   │    │   MCP Servers    │
                       │   (Chat + Tools) │    │  (Containerized) │
                       └──────────────────┘    └──────────────────┘
```

### Key Components

1. **Frontend (Port 3001)**: Web-based UI for interacting with MCP servers
2. **Backend (Port 8001)**: FastAPI server that connects to multi-mcp via HTTP
3. **Multi-MCP Server (Port 8081)**: Manages multiple MCP servers from [kfirtoledo/multi-mcp](https://github.com/kfirtoledo/multi-mcp)
4. **MCP Servers**: Individual protocol servers (filesystem, web search, GitHub, etc.)

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Azure OpenAI account with API access
- Optional: API keys for additional MCP servers (Brave Search, GitHub, etc.)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd multi-mcp-proxy
```

### 2. Configure Environment

Create a `.env` file with your credentials:

```env
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_ID=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Optional: API Keys for MCP Servers
BRAVE_API_KEY=your-brave-api-key        # For web search
GITHUB_TOKEN=your-github-token          # For GitHub operations
GRAFANA_API_KEY=your-grafana-key       # For Grafana dashboards
```

### 3. Deploy the Stack

```bash
# Make the deployment script executable
chmod +x deploy-stack.sh

# Deploy all services
./deploy-stack.sh
```

The script will:
- ✅ Check prerequisites
- ✅ Validate environment configuration
- ✅ Build Docker containers
- ✅ Start services in correct order
- ✅ Verify deployment health

### 4. Access the Application

Once deployed, access:
- **Web UI**: http://localhost:3001
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **Multi-MCP Server**: http://localhost:8081

## 📦 MCP Server Configuration

The system comes pre-configured with example MCP servers in `backend/mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "transport": "stdio",
      "description": "File system operations"
    },
    "web-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "transport": "stdio",
      "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
      "description": "Web search using Brave"
    }
  }
}
```

### Adding New MCP Servers

1. **Edit `backend/mcp.json`** to add your server configuration
2. **Restart the backend** to load the new configuration:
   ```bash
   docker-compose restart backend
   ```

### Available MCP Servers

- **Filesystem**: File and directory operations
- **Web Search**: Internet search via Brave API
- **GitHub**: Repository and issue management
- **Grafana**: Dashboard management
- **Prometheus**: Metrics querying
- [View more servers](https://github.com/modelcontextprotocol/servers)

## 🔧 Management Commands

### Stack Management

```bash
# Start the stack
./deploy-stack.sh

# Stop the stack
./deploy-stack.sh stop

# Restart services
./deploy-stack.sh restart

# Check status
./deploy-stack.sh status

# View logs
./deploy-stack.sh logs

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f multi-mcp-server
```

### Troubleshooting

```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
./deploy-stack.sh

# Check service health
curl http://localhost:8001/health
curl http://localhost:8081/health

# Test MCP tools
curl http://localhost:8001/api/tools

# Check registered servers
curl http://localhost:8001/api/servers
```

## 🐛 Common Issues & Solutions

### Issue: Network conflicts
```bash
# Clean up Docker networks
docker network prune -f
docker-compose down
./deploy-stack.sh
```

### Issue: Multi-MCP server not starting
```bash
# Check logs
docker-compose logs multi-mcp-server

# Rebuild the container
docker-compose build --no-cache multi-mcp-server
docker-compose up -d multi-mcp-server
```

### Issue: MCP servers not registering
```bash
# Verify mcp.json syntax
cat backend/mcp.json | python -m json.tool

# Check backend logs
docker-compose logs backend

# Restart backend
docker-compose restart backend
```

### Issue: Azure OpenAI not working
- Verify `.env` file has correct credentials
- Check API key permissions
- Confirm deployment name and API version
- Test with: `curl http://localhost:8001/api/llm/status`

## 🔒 Security Considerations

- **API Keys**: Store securely in `.env` file (never commit to Git)
- **Network**: Services communicate through Docker network isolation
- **CORS**: Currently allows all origins (restrict in production)
- **Authentication**: Not implemented (add for production use)

## 📚 Advanced Configuration

### Custom MCP Servers

Create custom MCP servers by following the [MCP specification](https://modelcontextprotocol.io):

```json
{
  "mcpServers": {
    "custom-server": {
      "command": "python",
      "args": ["path/to/your/server.py"],
      "transport": "stdio",
      "env": {
        "CUSTOM_VAR": "value"
      }
    }
  }
}
```

### Using Docker-based MCP Servers

For containerized MCP servers:

```json
{
  "mcpServers": {
    "docker-server": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--network", "mcp-network",
        "your-mcp-image:latest"
      ],
      "transport": "stdio"
    }
  }
}
```

### Environment Variable Substitution

The system supports environment variable substitution in `mcp.json`:
- `${VARIABLE_NAME}` will be replaced with the value from `.env`

## 🔄 Multi-MCP Endpoints Reference

This proxy integrates with the following Multi-MCP endpoints:

| Endpoint | Method | Description | 
|----------|--------|-------------|
| `/mcp_servers` | GET | List active MCP servers |
| `/mcp_servers` | POST | Add a new MCP server |
| `/mcp_servers/{name}` | DELETE | Remove an MCP server |
| `/mcp_tools` | GET | List all available tools with their server sources |
| `/tools/{tool_name}/call` | POST | Execute a specific tool |
| `/health` | GET | Check server health |

Tool names in multi-mcp are namespaced as `server_name::tool_name` to avoid conflicts when multiple servers provide tools with the same name.

## 💬 Chat Flow Architecture

The chat integration follows this sequence:

1. **User Input**: The user sends a message via the chat interface
2. **Tool Analysis**: The backend uses Azure OpenAI to analyze which tools might help answer the query
3. **Tool Selection**: The AI selects appropriate tools and parameters based on the query
4. **Tool Execution**: Selected tools are executed via the multi-mcp server
5. **Result Integration**: Tool results are collected and passed back to Azure OpenAI
6. **Response Generation**: The AI generates a final response incorporating tool results
7. **Response Display**: The answer is displayed to the user with information about which tools were used

The entire process is transparent to the user - they simply ask questions and get answers, with the system automatically handling tool selection and execution behind the scenes.

### Tool Selection Example

When a user asks "What's the weather in New York?", the system:

1. Analyzes the query to identify the intent (weather information)
2. Selects the appropriate tool (e.g., `weather::get_forecast`)
3. Determines parameters (location: "New York")
4. Executes the tool via multi-mcp
5. Returns a response that incorporates the weather data

This approach allows for powerful automation while maintaining a simple chat interface.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) by Anthropic
- [multi-mcp](https://github.com/kfirtoledo/multi-mcp) by kfirtoledo
- [MCP Servers](https://github.com/modelcontextprotocol/servers) community

## 📞 Support

For issues and questions:
- Check the [Troubleshooting](#-common-issues--solutions) section
- Review logs with `./deploy-stack.sh logs`
- Open an issue on GitHub

---

**Built with ❤️ using Model Context Protocol**