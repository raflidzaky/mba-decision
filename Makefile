# Config helper variable
PYTHON = python # Windows doesnt treat Python & Python3 the same way. Thus, it makes confusion if you wrote Python3
ENV = .venv
APP = source.api.api:app
UVICORN = uvicorn

# Install dependencies
install: 
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install -r requirements.txt

# Target to run test
test: source/training/test_load.py
	@echo "Running tests..."
	$(PYTHON) -m pytest $(TEST)

# Target to run the main app
run: 
	@echo "Running api.py..."
	$(UVICORN) $(APP)

# Target to start uvicorn server
serve: 
	@echo "Starting uvicorn server..."
	$(UVICORN) $(APP) --reload

# All in one command
all: install test run serve
	@echo "All tasks completed."