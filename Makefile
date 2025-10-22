help:
	@echo "ft_linear_regression"
	@echo ""
	@echo "Available targets:"
	@echo "  install      		- Install dependencies with UV"
	@echo "  clean        		- Clean virtual environment and cache"
	@echo "  run-training 		- Run the training module"
	@echo "  run-prediction 	- Run the prediction module"
	@echo "  help         		- Show this help message"

check-uv:
	@which uv > /dev/null || (echo "UV is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)

install: check-uv
	@echo "Installing dependencies with UV..."
	uv venv --clear
	uv pip install -e .
	@echo "Dependencies installed successfully!"

clean:
	@echo "Cleaning virtual environment and cache..."
	rm -rf .venv
	rm -rf __pycache__
	rm -rf src/**/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	@echo "Clean complete!"

run-training: check-uv
	@echo "Running training module..."
	@cd src/training && uv run python main.py

run-prediction: check-uv
	@echo "Running prediction module..."
	@cd src/prediction && uv run python estimate_price.py

setup: install
	@echo "Quick setup complete!"
	@echo ""
	@echo "Usage:"
	@echo "  make run-training    # Run training"
	@echo "  make run-prediction  # Run prediction"
	@echo "  make help           # Show all commands"

.PHONY: help install clean run-training run-prediction
