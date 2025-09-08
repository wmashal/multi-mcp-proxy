#!/bin/bash

# Multi-MCP Proxy Stack Deployment Script
# This script deploys the complete MCP proxy stack with proper multi-mcp integration

set -e

echo "🚀 Starting Multi-MCP Proxy Stack Deployment..."

# Configuration
COMPOSE_FILE="docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Check and create .env file
check_env_file() {
    print_status "Checking environment configuration..."

    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating template..."
        cat > .env << 'EOF'
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_DEPLOYMENT_ID=your-deployment-id
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Optional: API Keys for MCP Servers
BRAVE_API_KEY=your-brave-api-key
GITHUB_TOKEN=your-github-token
GRAFANA_API_KEY=your-grafana-api-key

# Multi-MCP is now integrated into the backend container
# No separate server needed
EOF
        print_warning "Please update the .env file with your actual credentials"
        print_warning "At minimum, configure Azure OpenAI settings"
        exit 1
    fi

    # Check if Azure OpenAI is configured
    source .env
    if [ -z "$AZURE_OPENAI_ENDPOINT" ] || [ "$AZURE_OPENAI_ENDPOINT" == "https://your-resource.openai.azure.com/" ]; then
        print_error "Azure OpenAI endpoint not configured in .env file"
        print_error "Please update AZURE_OPENAI_ENDPOINT with your actual endpoint"
        exit 1
    fi

    if [ -z "$AZURE_OPENAI_API_KEY" ] || [ "$AZURE_OPENAI_API_KEY" == "your-azure-api-key" ]; then
        print_error "Azure OpenAI API key not configured in .env file"
        print_error "Please update AZURE_OPENAI_API_KEY with your actual API key"
        exit 1
    fi

    print_success "Environment configuration validated"
}

# Stop existing containers
cleanup_existing() {
    print_status "Cleaning up existing containers..."

    # Stop containers
    docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || true

    # Remove any orphaned MCP server containers
    docker ps -a | grep "mcp-" | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true

    print_success "Cleanup completed"
}

# Build and deploy the stack
deploy_stack() {
    print_status "Building and starting the Multi-MCP Proxy stack..."

    # Build all services
    print_status "Building containers..."
    docker-compose -f $COMPOSE_FILE build

    # Start services (backend now includes integrated Multi-MCP server)
    # Start backend
    print_status "Starting MCP Proxy Backend..."
    docker-compose -f $COMPOSE_FILE up -d backend

    # Wait for backend to be healthy
    print_status "Waiting for Backend to be ready..."
    counter=0
    while [ $counter -lt $timeout ]; do
        if docker-compose -f $COMPOSE_FILE ps backend 2>/dev/null | grep -q "healthy\|running"; then
            print_success "Backend is ready"
            break
        fi
        sleep 2
        counter=$((counter + 2))
        echo -n "."
    done
    echo ""

    if [ $counter -ge $timeout ]; then
        print_error "Backend failed to start within $timeout seconds"
        print_status "Checking logs..."
        docker-compose -f $COMPOSE_FILE logs --tail=50 backend
        exit 1
    fi

    # Start frontend
    print_status "Starting Frontend..."
    docker-compose -f $COMPOSE_FILE up -d frontend

    print_success "All services started successfully"
}

# Verify deployment
verify_deployment() {
    print_status "Verifying deployment..."

    # Multi-MCP is now integrated into the backend container
    # Health check is handled by backend service

    # Check Backend API health
    if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
        health_data=$(curl -s http://localhost:8001/health)
        print_success "Backend API is responding"
        echo "  Health status: $health_data"
    else
        print_warning "Backend API health check failed"
    fi

    # Check Frontend
    if curl -s -f http://localhost:3001 > /dev/null 2>&1; then
        print_success "Frontend is accessible"
    else
        print_warning "Frontend is not accessible"
    fi
}

# Display status and URLs
show_status() {
    print_status "Deployment Summary"
    echo ""
    echo "📊 Service Status:"
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    echo "🌐 Access URLs:"
    echo "  Frontend UI:      http://localhost:3001"
    echo "  Backend API:      http://localhost:8001"
    echo "  API Docs:         http://localhost:8001/docs"
    echo "  Multi-MCP:        Integrated into Backend"
    echo ""
    echo "📋 Useful Commands:"
    echo "  View all logs:        docker-compose -f $COMPOSE_FILE logs -f"
    echo "  View backend logs:    docker-compose -f $COMPOSE_FILE logs -f backend"
    echo "  View all backend logs: docker-compose -f $COMPOSE_FILE logs -f backend"
    echo "  Stop stack:           docker-compose -f $COMPOSE_FILE down"
    echo "  Restart services:     docker-compose -f $COMPOSE_FILE restart"
    echo ""
    echo "🔧 Configuration:"
    echo "  MCP servers config:   backend/mcp.json"
    echo "  Environment vars:     .env"
    echo ""
    echo "📚 API Endpoints:"
    echo "  GET  /api/servers     - List all MCP servers"
    echo "  GET  /api/tools       - List available tools"
    echo "  POST /api/chat        - Chat with AI using MCP tools"
    echo "  GET  /api/llm/status  - Check LLM configuration"
    echo ""
}

# Main execution
main() {
    print_status "Multi-MCP Proxy Stack Deployment"
    echo ""

    check_prerequisites
    check_env_file
    cleanup_existing
    deploy_stack
    
    # Wait a bit for services to fully initialize
    sleep 5
    
    verify_deployment
    show_status

    print_success "Deployment completed successfully! 🚀"
    print_status "You can now access the UI at http://localhost:3001"
}

# Handle script arguments
case "${1:-}" in
    stop)
        print_status "Stopping Multi-MCP Proxy stack..."
        docker-compose -f $COMPOSE_FILE down
        print_success "Stack stopped"
        ;;
    restart)
        print_status "Restarting Multi-MCP Proxy stack..."
        docker-compose -f $COMPOSE_FILE restart
        print_success "Stack restarted"
        ;;
    logs)
        docker-compose -f $COMPOSE_FILE logs -f ${2:-}
        ;;
    status)
        verify_deployment
        show_status
        ;;
    *)
        main "$@"
        ;;
esac