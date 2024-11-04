@echo off
echo Checking Docker status...
docker info > nul 2>&1
if errorlevel 1 (
    echo Docker is not running. Please start Docker and try again.
    exit /b 1
)

echo Creating data directory...
if not exist data mkdir data

echo Cleaning up Docker resources...
docker-compose down
docker system prune -f

echo Building and starting containers...
docker-compose up --build -d

echo Checking container status...
docker ps | find "chatbot-app" > nul
if errorlevel 1 (
    echo Failed to start container. Checking logs...
    docker-compose logs
) else (
    echo Application is running!
    echo Access the application at: http://localhost:8501
    echo To view logs: docker-compose logs -f
    echo To stop: docker-compose down
)