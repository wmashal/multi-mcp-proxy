import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Import LangChain components
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool

# Import MCP client
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp_proxy_backend")

app = FastAPI(title="MCP Proxy Backend")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API requests
class MCPServerConfig(BaseModel):
    name: str
    command: Optional[str] = None
    args: Optional[List[str]] = []
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = {}
    transport: Optional[str] = "stdio"

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ServerStatus(BaseModel):
    name: str
    status: str  # "connected", "disconnected", "error"
    capabilities: Optional[Dict] = None
    error_message: Optional[str] = None
    config: Optional[Dict] = None

# Global state using the official multi-mcp client
class MCPProxyManager:
    def __init__(self):
        self.mcp_client = None
        self.azure_llm = None
        self.agent = None
        self.connected_servers = {}
        self.config_file_path = "mcp.json"
        self.multi_mcp_session = None

    async def initialize(self):
        """Initialize the MCP client and Azure OpenAI"""
        try:
            # Initialize Azure OpenAI
            self.azure_llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_ID"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                temperature=0.7,
            )

            # Load and connect to configured servers
            await self.load_and_connect_servers()

            logger.info("✅ MCP Proxy Manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP Proxy Manager: {e}")
            return False

    async def load_and_connect_servers(self):
        """Spawn and connect to the Multi-MCP server using langchain-mcp-adapters"""
        try:
            logger.info("Starting Multi-MCP server as subprocess...")
            
            # Create the MCP client without initial connections
            self.mcp_client = MultiServerMCPClient()
            
            # Connect to the Multi-MCP server by spawning it as a subprocess
            # The Multi-MCP main.py should be available from the cloned repo
            multi_mcp_path = "/app/multi-mcp/main.py"
            mcp_config_path = "/app/mcp.json"
            
            # First, let's check if the Multi-MCP files exist
            logger.debug(f"Checking Multi-MCP path: {multi_mcp_path}")
            if not os.path.exists(multi_mcp_path):
                logger.error(f"Multi-MCP main.py not found at {multi_mcp_path}")
                # Let's list what's actually in the multi-mcp directory
                multi_mcp_dir = "/app/multi-mcp"
                if os.path.exists(multi_mcp_dir):
                    logger.debug(f"Contents of {multi_mcp_dir}: {os.listdir(multi_mcp_dir)}")
                else:
                    logger.error(f"Multi-MCP directory {multi_mcp_dir} does not exist")
                raise FileNotFoundError(f"Multi-MCP main.py not found at {multi_mcp_path}")
            else:
                logger.debug(f"✓ Multi-MCP main.py found at {multi_mcp_path}")
            
            logger.debug(f"Checking MCP config path: {mcp_config_path}")
            if not os.path.exists(mcp_config_path):
                logger.error(f"MCP config not found at {mcp_config_path}")
                # Let's list what's in /app
                if os.path.exists("/app"):
                    logger.debug(f"Contents of /app: {os.listdir('/app')}")
                raise FileNotFoundError(f"MCP config not found at {mcp_config_path}")
            else:
                logger.debug(f"✓ MCP config found at {mcp_config_path}")
            
            # Add the Multi-MCP server as a connection to the client
            connections = {
                "multi-mcp": {
                    "command": "python",
                    "args": [multi_mcp_path, "--transport", "stdio", "--config", mcp_config_path],
                    "transport": "stdio",
                    "env": {
                        "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", ""),
                        "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", ""),
                        "GRAFANA_API_KEY": os.getenv("GRAFANA_API_KEY", ""),
                        "PYTHONPATH": "/app/multi-mcp",
                        "PATH": os.getenv("PATH", "")
                    }
                }
            }
            
            # Reinitialize MCP client with the Multi-MCP connection
            logger.debug(f"Creating MultiServerMCPClient with connections: {connections}")
            self.mcp_client = MultiServerMCPClient(connections)
            
            # Debug the client after creation
            logger.debug(f"MCP client connections: {self.mcp_client.connections}")
            logger.debug(f"MCP client connections keys: {list(self.mcp_client.connections.keys())}")
            
            self.connected_servers["multi-mcp"] = connections["multi-mcp"]
            
            logger.info("✅ Multi-MCP client initialized with connection")
            
            # Create the agent
            await self.create_agent()
            
        except Exception as e:
            logger.error(f"Failed to initialize Multi-MCP: {e}")
            logger.exception("Full error details:")
            self.mcp_client = None

    async def connect_to_server(self, name: str, config: Dict):
        """Add a new server to the Multi-MCP server via API"""
        try:
            logger.info(f"Adding MCP server '{name}' to Multi-MCP server...")
            
            # Get Multi-MCP server connection details
            multi_mcp_host = os.getenv("MULTI_MCP_HOST", "multi-mcp-server")
            multi_mcp_port = os.getenv("MULTI_MCP_PORT", "8081")
            multi_mcp_api_url = f"http://{multi_mcp_host}:{multi_mcp_port}"
            
            # Add server to Multi-MCP via its API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{multi_mcp_api_url}/mcp_servers",
                    json={"mcpServers": {name: config}}
                )
                
                if response.status_code == 200:
                    self.connected_servers[name] = config
                    logger.info(f"✅ Successfully added MCP server '{name}' to Multi-MCP")
                    
                    # Recreate the agent with updated tools
                    await self.create_agent()
                else:
                    raise Exception(f"Failed to add server to Multi-MCP: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"❌ Failed to add server '{name}': {e}")
            raise

    async def create_agent(self):
        """Create or recreate the LangGraph agent with current tools"""
        try:
            if self.mcp_client and self.azure_llm:
                logger.info("Getting tools from Multi-MCP server...")
                
                try:
                    tools = await self.mcp_client.get_tools()
                    logger.info(f"🔧 Creating agent with {len(tools)} tools from Multi-MCP")
                    
                    if tools:
                        tool_names = [tool.name for tool in tools if hasattr(tool, 'name')]
                        logger.info(f"Available tools: {tool_names}")
                        
                        self.agent = create_react_agent(self.azure_llm, tools)
                        logger.info("✅ Agent created successfully with Multi-MCP tools")
                    else:
                        logger.warning("No tools available from Multi-MCP, creating agent without tools")
                        self.agent = create_react_agent(self.azure_llm, [])
                        
                except Exception as tool_error:
                    logger.error(f"Error getting tools from Multi-MCP: {tool_error}")
                    logger.exception("Tool error details:")
                    
                    # Create agent without tools as fallback
                    logger.info("Creating agent without tools as fallback")
                    self.agent = create_react_agent(self.azure_llm, [])
                    
            elif self.azure_llm:
                logger.warning("Multi-MCP client not available, creating agent without tools")
                self.agent = create_react_agent(self.azure_llm, [])
            else:
                logger.warning("Azure LLM not available")
                
        except Exception as e:
            logger.error(f"❌ Failed to create agent: {e}")
            logger.exception("Agent creation error details:")

    async def get_tools(self):
        """Get all available tools from Multi-MCP server"""
        if self.mcp_client:
            try:
                logger.debug(f"Getting tools from Multi-MCP client (type: {type(self.mcp_client)})...")
                logger.debug(f"Client connections: {self.mcp_client.connections}")
                logger.debug(f"Number of connections: {len(self.mcp_client.connections)}")
                
                # Check if connections is empty - this might be the issue
                if not self.mcp_client.connections:
                    logger.error("No connections in MCP client - this explains the list return")
                    return []
                
                # Check if mcp_client is actually a MultiServerMCPClient instance
                if hasattr(self.mcp_client, 'get_tools'):
                    # Check if get_tools is a coroutine or returns a list directly
                    get_tools_method = getattr(self.mcp_client, 'get_tools')
                    logger.debug(f"get_tools method type: {type(get_tools_method)}")
                    
                    # Try to inspect the method signature
                    import inspect
                    if inspect.iscoroutinefunction(get_tools_method):
                        logger.debug("get_tools is a coroutine function, awaiting it")
                        tools = await self.mcp_client.get_tools()
                    else:
                        logger.debug("get_tools is not a coroutine function, calling directly")
                        tools = self.mcp_client.get_tools()
                        
                    logger.debug(f"Successfully got {len(tools)} tools from Multi-MCP")
                    
                    if len(tools) == 0:
                        logger.warning("No tools received from Multi-MCP server subprocess")
                        logger.debug("This could mean:")
                        logger.debug("1. Multi-MCP server subprocess is not starting")
                        logger.debug("2. Multi-MCP server has no tools configured") 
                        logger.debug("3. Connection to subprocess failed")
                        
                        # Let's try to debug the MCP client version and methods
                        logger.info("Attempting to debug Multi-MCP client version and methods...")
                        try:
                            # Check what methods are available
                            logger.debug(f"Available methods on MCP client: {dir(self.mcp_client)}")
                            
                            # Check langchain-mcp-adapters version
                            import langchain_mcp_adapters
                            logger.debug(f"langchain-mcp-adapters version: {getattr(langchain_mcp_adapters, '__version__', 'unknown')}")
                            
                            # Try to inspect the get_tools method signature
                            import inspect
                            get_tools_sig = inspect.signature(self.mcp_client.get_tools)
                            logger.debug(f"get_tools method signature: {get_tools_sig}")
                            
                            # Check if there are any sessions or connections active
                            if hasattr(self.mcp_client, 'connections'):
                                logger.debug(f"Connections available: {list(self.mcp_client.connections.keys())}")
                                
                            # Check the server_name_to_tools mapping
                            if hasattr(self.mcp_client, 'server_name_to_tools'):
                                logger.debug(f"server_name_to_tools: {self.mcp_client.server_name_to_tools}")
                                logger.debug(f"server_name_to_tools keys: {list(self.mcp_client.server_name_to_tools.keys())}")
                                
                                # Check if multi-mcp server has tools mapped
                                if 'multi-mcp' in self.mcp_client.server_name_to_tools:
                                    multi_mcp_tools = self.mcp_client.server_name_to_tools['multi-mcp']
                                    logger.debug(f"Multi-MCP server has {len(multi_mcp_tools)} tools directly mapped")
                                    if len(multi_mcp_tools) > 0:
                                        logger.info(f"SUCCESS: Found {len(multi_mcp_tools)} tools in server_name_to_tools mapping!")
                                        tools = multi_mcp_tools
                                        
                                        # IMPORTANT: Recreate the agent with the loaded tools
                                        logger.info(f"🔄 Recreating agent with {len(tools)} MCP tools...")
                                        if self.azure_llm:
                                            self.agent = create_react_agent(self.azure_llm, tools)
                                            logger.info("✅ Agent recreated successfully with MCP tools!")
                                else:
                                    logger.debug("Multi-MCP server not found in server_name_to_tools mapping")
                                    logger.info("Attempting to manually connect to Multi-MCP server...")
                                    try:
                                        # Try to explicitly connect to the server with full connection details
                                        if hasattr(self.mcp_client, 'connect_to_server'):
                                            connection_config = self.mcp_client.connections.get('multi-mcp', {})
                                            logger.debug(f"Using connection config: {connection_config}")
                                            
                                            await self.mcp_client.connect_to_server(
                                                'multi-mcp',
                                                command=connection_config.get('command'),
                                                args=connection_config.get('args', []),
                                                env=connection_config.get('env', {})
                                            )
                                            logger.debug("Manual connection attempt completed")
                                            
                                            # Check tools again after connection
                                            updated_tools = self.mcp_client.get_tools()
                                            logger.debug(f"Tools after manual connection: {len(updated_tools)}")
                                            if len(updated_tools) > 0:
                                                logger.info(f"SUCCESS: Got {len(updated_tools)} tools after manual connection!")
                                                tools = updated_tools
                                                
                                                # IMPORTANT: Recreate the agent with the loaded tools
                                                logger.info(f"🔄 Recreating agent with {len(tools)} MCP tools...")
                                                if self.azure_llm:
                                                    self.agent = create_react_agent(self.azure_llm, tools)
                                                    logger.info("✅ Agent recreated successfully with MCP tools!")
                                    except Exception as connect_error:
                                        logger.error(f"Manual connection error: {connect_error}")
                                        logger.exception("Manual connection error details")
                                    
                            # Check if there are active sessions
                            if hasattr(self.mcp_client, 'sessions'):
                                logger.debug(f"Active sessions: {self.mcp_client.sessions}")
                                logger.debug(f"Session keys: {list(self.mcp_client.sessions.keys()) if self.mcp_client.sessions else 'None'}")
                                
                            # Maybe the tools are loaded lazily when sessions are created?
                            if hasattr(self.mcp_client, 'session'):
                                logger.debug("Client has session method - trying to use session context")
                                try:
                                    # Try to get tools using session context for the multi-mcp server
                                    async with self.mcp_client.session("multi-mcp") as session:
                                        from langchain_mcp_adapters.tools import load_mcp_tools
                                        session_tools = await load_mcp_tools(session)
                                        logger.debug(f"Got {len(session_tools)} tools from session context")
                                        if len(session_tools) > 0:
                                            logger.info(f"SUCCESS: Found {len(session_tools)} tools using session context!")
                                            # Update our tools list
                                            tools = session_tools
                                            
                                            # IMPORTANT: Recreate the agent with the loaded tools
                                            logger.info(f"🔄 Recreating agent with {len(tools)} MCP tools...")
                                            if self.azure_llm:
                                                self.agent = create_react_agent(self.azure_llm, tools)
                                                logger.info("✅ Agent recreated successfully with MCP tools!")
                                except Exception as session_error:
                                    logger.error(f"Session error: {session_error}")
                                    logger.exception("Session error details")
                                
                        except Exception as debug_error:
                            logger.error(f"Debug error: {debug_error}")
                            logger.exception("Debug error details")
                    
                    # Return the tools (might have been updated in debug section)
                    return tools
                else:
                    logger.error(f"mcp_client does not have get_tools method. Type: {type(self.mcp_client)}")
                    return []
                    
            except TypeError as te:
                logger.error(f"TypeError getting tools from Multi-MCP: {te}")
                logger.error(f"mcp_client type: {type(self.mcp_client)}")
                logger.error(f"mcp_client value: {self.mcp_client}")
                return []
            except Exception as e:
                logger.error(f"Failed to get tools from Multi-MCP: {e}")
                logger.exception("Tools error details:")
                return []
        else:
            logger.warning("Multi-MCP client not available")
            return []

    async def execute_query(self, user_query: str, custom_prompt: str = None):
        """Execute a query using the LangGraph agent with custom prompt"""
        if not self.agent:
            raise Exception("Agent not initialized")

        try:
            # Prepare the message with custom prompt if provided
            if custom_prompt:
                system_message = {"role": "system", "content": custom_prompt}
                user_message = {"role": "user", "content": user_query}
                messages = [system_message, user_message]
            else:
                messages = [{"role": "user", "content": user_query}]

            # Execute the query using the agent
            response = await self.agent.ainvoke({"messages": messages})

            # Extract the final response and tool results
            final_message = response['messages'][-1] if response['messages'] else None

            # Extract tool calls from the conversation
            tool_results = []
            executed_tools = []

            for msg in response['messages']:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_results.append({
                            "tool": tool_call['name'],
                            "parameters": tool_call.get('args', {}),
                            "result": "Tool executed via agent"
                        })
                        executed_tools.append(tool_call['name'])

            return {
                "response": final_message.content if final_message else "No response generated",
                "tool_results": tool_results,
                "executed_tools": executed_tools,
                "full_conversation": response['messages']
            }

        except Exception as e:
            logger.error(f"❌ Query execution failed: {e}")
            raise

    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        if self.mcp_client:
            # The MultiServerMCPClient handles disconnection automatically
            # when used as a context manager or when the object is destroyed
            self.connected_servers.clear()
            self.agent = None
            logger.info("🔌 Disconnected from all MCP servers")

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server name that provides a specific tool"""
        if not self.mcp_client:
            return None

        # Check if the MultiServerMCPClient has server-specific tool information
        try:
            # The MultiServerMCPClient should ideally provide this mapping
            # For now, we'll iterate through connected servers to find which one provides the tool
            for server_name in self.connected_servers.keys():
                # This is a simplified approach - the actual implementation would depend
                # on how the MultiServerMCPClient exposes server-specific tool information
                tools = self.get_tools()
                for tool in tools:
                    if tool.name == tool_name:
                        # Try to get server info from tool if available
                        if hasattr(tool, 'server_name'):
                            return tool.server_name
                        # Fallback to the first connected server for now
                        # In practice, the MultiServerMCPClient should provide this mapping
                        return server_name
        except Exception as e:
            logger.warning(f"Could not determine server for tool {tool_name}: {e}")

        return None

    async def get_tool_to_server_mapping(self) -> Dict[str, str]:
        """Get a mapping of tool names to their source servers"""
        mapping = {}
        if not self.mcp_client:
            return mapping

        try:
            tools = await self.get_tools()
            for tool in tools:
                # Try to get server info from tool metadata
                if hasattr(tool, 'server_name'):
                    mapping[tool.name] = tool.server_name
                elif hasattr(tool, 'metadata') and isinstance(tool.metadata, dict):
                    server = tool.metadata.get('server') or tool.metadata.get('source')
                    if server:
                        mapping[tool.name] = server
                else:
                    # Fallback: assign to first available server
                    # This is not ideal but works when server info is not available
                    if self.connected_servers:
                        mapping[tool.name] = list(self.connected_servers.keys())[0]
        except Exception as e:
            logger.warning(f"Could not build tool-to-server mapping: {e}")

        return mapping

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]):
        """Execute a specific MCP tool via Multi-MCP"""
        if not self.mcp_client:
            raise Exception("Multi-MCP client not initialized")

        try:
            # Get the tools and find the one to call
            tools = await self.get_tools()
            tool_to_call = None
            
            for tool in tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    tool_to_call = tool
                    break
                    
            if not tool_to_call:
                raise Exception(f"Tool '{tool_name}' not found in Multi-MCP server")
                
            # Call the tool with parameters
            logger.info(f"Calling Multi-MCP tool '{tool_name}' with parameters: {parameters}")
            result = await tool_to_call.ainvoke(parameters)
            logger.info(f"Tool '{tool_name}' executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise

# Global proxy manager instance
proxy_manager = MCPProxyManager()
config_file_path = "mcp.json"

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Load MCP server configurations including custom prompts
def load_mcp_config():
    """Load MCP server configurations from mcp.json"""
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        return config.get('mcpServers', {})
    except Exception as e:
        logger.error(f"Error loading MCP config: {e}")
        return {}

async def get_custom_prompt_for_tools(executed_tools):
    """Get appropriate custom prompt based on executed tools"""
    mcp_config = load_mcp_config()

    # Use the dynamic tool-to-server mapping from the proxy manager
    tool_to_server_mapping = await proxy_manager.get_tool_to_server_mapping()

    # Determine which MCP servers were used
    used_servers = set()
    for tool in executed_tools:
        server = tool_to_server_mapping.get(tool)
        if server:
            used_servers.add(server)

    # If multiple servers were used, combine their prompts
    if len(used_servers) > 1:
        combined_prompts = []
        for server in used_servers:
            server_config = mcp_config.get(server, {})
            custom_prompt = server_config.get('customPrompt')
            if custom_prompt:
                combined_prompts.append(f"For {server} operations: {custom_prompt}")

        if combined_prompts:
            return "You are a multi-tool assistant with specialized knowledge in multiple domains. " + " ".join(combined_prompts) + " Use the tool results to provide a comprehensive answer that leverages your expertise in all relevant domains."

    # If only one server was used, use its specific prompt
    elif len(used_servers) == 1:
        server = list(used_servers)[0]
        server_config = mcp_config.get(server, {})
        custom_prompt = server_config.get('customPrompt')
        if custom_prompt:
            return custom_prompt + " Use the tool results to answer the user's question comprehensively."

    # Default fallback prompt
    return "You are a helpful assistant. Use the available tools to answer the user's question comprehensively."

# Health endpoint for Docker healthcheck
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker container."""
    multi_mcp_healthy = bool(proxy_manager.connected_servers)

    try:
        available_tools = await proxy_manager.get_tools()
        tools_count = len(available_tools)
    except Exception as e:
        logger.warning(f"Could not get tools count: {e}")
        tools_count = 0

    return {
        "status": "healthy",
        "multi_mcp_connected": multi_mcp_healthy,
        "available_tools": tools_count,
        "registered_servers": len(proxy_manager.connected_servers),
        "azure_configured": bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)
    }

# Config management functions
def load_mcp_config(path: str = None) -> Dict:
    """Load MCP configuration from JSON file."""
    if path is None:
        path = config_file_path

    if not os.path.exists(path):
        logger.info(f"Config file {path} not found, creating empty config")
        empty_config = {"mcpServers": {}}
        save_mcp_config(empty_config, path)
        return empty_config

    try:
        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file)
            logger.info(f"Loaded MCP config from {path}")
            
            # Fix Docker networking issues in config
            if "mcpServers" in config:
                for server_name, server_config in config["mcpServers"].items():
                    # Replace host.docker.internal with proper Docker network names
                    if "args" in server_config:
                        server_config["args"] = [
                            arg.replace("host.docker.internal", "host.docker.internal" if os.getenv("DOCKER_ENV") else "localhost")
                            for arg in server_config["args"]
                        ]
                    # Remove host network mode for Docker containers in our stack
                    if "args" in server_config and "--network" in server_config["args"]:
                        try:
                            idx = server_config["args"].index("--network")
                            if idx + 1 < len(server_config["args"]) and server_config["args"][idx + 1] == "host":
                                # Remove both --network and host
                                server_config["args"].pop(idx)
                                server_config["args"].pop(idx)
                        except:
                            pass
            
            return config
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON config: {e}")
        return {"mcpServers": {}}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {"mcpServers": {}}

def save_mcp_config(config: Dict, path: str = None) -> bool:
    """Save MCP configuration to JSON file."""
    if path is None:
        path = config_file_path

    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            logger.info(f"Saved MCP config to {path}")
            return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

async def call_azure_openai(messages: List[Dict], max_tokens: int = 500, temperature: float = 0.1) -> str:
    """Call Azure OpenAI API directly."""
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Azure OpenAI configuration missing")

    try:
        azure_url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT_ID}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                azure_url,
                headers={
                    'Content-Type': 'application/json',
                    'api-key': AZURE_OPENAI_API_KEY
                },
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"API error: {response.status_code} - {response.text}")

            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    except Exception as error:
        logger.error(f"Azure OpenAI call failed: {error}")
        raise HTTPException(status_code=500, detail=f"Azure OpenAI call failed: {str(error)}")

async def initialize_mcp_client():
    """Initialize connection to the Multi-MCP server."""
    global proxy_manager

    try:
        # Initialize the proxy manager
        await proxy_manager.initialize()
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Multi-MCP client: {e}")
        return False

async def initialize_proxy():
    """Initialize the MCP proxy connection."""
    logger.info(f"Initializing connection to Multi-MCP server...")

    success = await initialize_mcp_client()
    if success:
        logger.info("✅ Successfully initialized MCP proxy")
    else:
        logger.warning("⚠️ Failed to fully initialize MCP proxy. Some features may not work.")

@app.on_event("startup")
async def startup_event():
    """Initialize the proxy on server startup."""
    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        logger.info("✅ Azure OpenAI configured")
    else:
        logger.warning("⚠️ Azure OpenAI configuration missing. Chat features will be limited.")
    
    await initialize_proxy()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    global proxy_manager
    await proxy_manager.disconnect_all()
    proxy_manager = MCPProxyManager()  # Reset to new instance
    logger.info("MCP client connections cleaned up")

@app.get("/api/config")
async def get_config():
    """Get the current MCP configuration."""
    config = load_mcp_config()
    return config

@app.post("/api/config")
async def update_config(config: Dict):
    """Update the entire MCP configuration."""
    try:
        # Validate config structure
        if "mcpServers" not in config:
            raise HTTPException(status_code=400, detail="Config must contain 'mcpServers' object")

        # Save config
        if not save_mcp_config(config):
            raise HTTPException(status_code=500, detail="Failed to save configuration")

        # Restart proxy with new config
        await initialize_proxy()

        return {"message": "Configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/servers", response_model=List[ServerStatus])
async def get_servers():
    """Get status of all registered MCP servers."""
    config = load_mcp_config()
    servers = []

    # Initialize proxy manager if not already done
    if not proxy_manager.mcp_client:
        await proxy_manager.initialize()

    # Get active servers from the Multi-MCP server
    try:
        multi_mcp_host = os.getenv("MULTI_MCP_HOST", "multi-mcp-server")
        multi_mcp_port = os.getenv("MULTI_MCP_PORT", "8081")
        multi_mcp_api_url = f"http://{multi_mcp_host}:{multi_mcp_port}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{multi_mcp_api_url}/mcp_servers", timeout=10.0)
            if response.status_code == 200:
                active_servers_data = response.json()
                active_server_names = set(active_servers_data.keys()) if isinstance(active_servers_data, dict) else set()
                logger.info(f"Active servers from Multi-MCP: {active_server_names}")
            else:
                logger.warning(f"Could not get servers from Multi-MCP: {response.status_code}")
                active_server_names = set()
    except Exception as e:
        logger.error(f"Error getting servers from Multi-MCP: {e}")
        active_server_names = set()

    # Build server status list
    for name, server_config in config.get("mcpServers", {}).items():
        # Check if server has tools available (meaning it's connected)
        server_tools = []
        
        try:
            available_tools = await proxy_manager.get_tools()
            # Filter tools for this server (if we can determine which server they come from)
            for tool in available_tools:
                if hasattr(tool, 'server_name') and tool.server_name == name:
                    server_tools.append({
                        "name": tool.name,
                        "description": tool.description if hasattr(tool, 'description') else f"{tool.name} from {name} server"
                    })
                elif hasattr(tool, 'name') and name in proxy_manager.connected_servers:
                    # Fallback: assume tools belong to connected servers
                    server_tools.append({
                        "name": tool.name,
                        "description": tool.description if hasattr(tool, 'description') else f"{tool.name} from {name} server"
                    })
                    
            status = "connected" if name in proxy_manager.connected_servers else "configured"
        except Exception as e:
            logger.error(f"Error getting tools for server {name}: {e}")
            status = "configured"

        servers.append(
            ServerStatus(
                name=name,
                status=status,
                capabilities={
                    "tools": server_tools,
                    "resources": [],
                    "prompts": []
                },
                error_message=None,
                config=server_config
            )
        )

    return servers

@app.post("/api/servers", response_model=ServerStatus)
async def add_server(server_config: MCPServerConfig):
    """Add a new MCP server to the configuration."""
    name = server_config.name

    # Load current config
    config = load_mcp_config()

    # Check if server already exists
    if name in config.get("mcpServers", {}):
        raise HTTPException(status_code=400, detail=f"Server '{name}' already exists")

    # Create server config
    new_server_config = {
        "transport": server_config.transport or "stdio"
    }

    if server_config.command:
        new_server_config["command"] = server_config.command
        new_server_config["args"] = server_config.args or []
        new_server_config["env"] = server_config.env or {}
    elif server_config.url:
        new_server_config["url"] = server_config.url
        new_server_config["env"] = server_config.env or {}
        new_server_config["transport"] = "sse"
    else:
        raise HTTPException(status_code=400, detail="Either 'command' or 'url' must be provided")

    # Add to config
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    config["mcpServers"][name] = new_server_config

    # Save config
    if not save_mcp_config(config):
        raise HTTPException(status_code=500, detail="Failed to save configuration")

    # Register with multi-mcp
    try:
        await proxy_manager.connect_to_server(
            name=name,
            config=new_server_config
        )

        logger.info(f"✅ Server '{name}' registered with multi-mcp")
    except Exception as e:
        logger.error(f"Error registering server with multi-mcp: {e}")

    return ServerStatus(
        name=name,
        status="registered" if name in proxy_manager.connected_servers else "configured",
        capabilities=None,
        config=new_server_config
    )

@app.delete("/api/servers/{server_name}")
async def remove_server(server_name: str):
    """Remove an MCP server from the configuration."""
    # Load current config
    config = load_mcp_config()

    # Check if server exists
    if server_name not in config.get("mcpServers", {}):
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    # Remove from proxy manager if connected
    if server_name in proxy_manager.connected_servers:
        try:
            # Remove server from connections
            if proxy_manager.mcp_client and server_name in proxy_manager.mcp_client.connections:
                del proxy_manager.mcp_client.connections[server_name]
                
            del proxy_manager.connected_servers[server_name]
            
            # Recreate agent with remaining tools
            await proxy_manager.create_agent()
            
            logger.info(f"Removed server '{server_name}' from proxy manager")
        except Exception as e:
            logger.error(f"Error removing server from proxy manager: {e}")

    # Remove from config
    del config["mcpServers"][server_name]

    # Save config
    if not save_mcp_config(config):
        raise HTTPException(status_code=500, detail="Failed to save configuration")

    return {"message": f"Server '{server_name}' removed successfully"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()

    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back or handle specific commands
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/api/tools")
async def get_tools():
    """Get all available MCP tools."""
    return await proxy_manager.get_tools()

@app.post("/api/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, parameters: Dict[str, Any] = None):
    """Execute a specific MCP tool."""
    if not proxy_manager.connected_servers:
        raise HTTPException(status_code=503, detail="MCP client not initialized")

    try:
        # Execute tool via multi-mcp client
        result = await proxy_manager.call_tool(tool_name, parameters or {})
        return result
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint that uses MCP tools and Azure OpenAI via LangGraph agent."""
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        raise HTTPException(status_code=503, detail="Azure OpenAI not configured")

    try:
        # Get the user's latest message
        user_message = request.messages[-1].content if request.messages else ""

        # Initialize proxy manager if not already done
        if not proxy_manager.agent:
            await proxy_manager.initialize()

        if not proxy_manager.agent:
            raise HTTPException(status_code=503, detail="MCP agent not initialized")

        # Get custom prompt based on available tools (pre-determine servers that might be used)
        available_tools = await proxy_manager.get_tools()
        tool_names = [tool.name for tool in available_tools]
        custom_prompt = await get_custom_prompt_for_tools(tool_names)

        # Execute query using the LangGraph agent with custom prompt
        result = await proxy_manager.execute_query(user_message, custom_prompt)

        return {
            "message": result["response"],
            "model": request.model or AZURE_OPENAI_DEPLOYMENT_ID,
            "tool_results": result["tool_results"],
            "tools_executed": len(result["tool_results"]),
            "executed_tools": result["executed_tools"]
        }

    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        # Fallback to direct Azure OpenAI call
        try:
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            response = await call_azure_openai(openai_messages, max_tokens=1000, temperature=0.7)
            return {
                "message": response,
                "model": request.model or AZURE_OPENAI_DEPLOYMENT_ID,
                "tool_results": [],
                "tools_executed": 0,
                "executed_tools": []
            }
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/llm/status")
async def get_llm_status():
    """Get current LLM configuration and status."""
    try:
        available_tools = await proxy_manager.get_tools()
        tools_count = len(available_tools)
    except Exception as e:
        logger.warning(f"Could not get tools count: {e}")
        tools_count = 0

    return {
        "enabled": bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT),
        "provider": "azure",
        "deployment": AZURE_OPENAI_DEPLOYMENT_ID,
        "multi_mcp_connected": bool(proxy_manager.connected_servers),
        "tools_available": tools_count,
        "servers_registered": len(proxy_manager.connected_servers)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
