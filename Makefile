# Config helper variable
PYTHON = python # Windows doesnt treat Python & Python3 the same way. Thus, it makes confusion if you wrote Python3
ENV = .venv

# Target to run test
test: test_load.py
	@echo "Running tests..."
	$(PYTHON) -m pytest $(TEST)